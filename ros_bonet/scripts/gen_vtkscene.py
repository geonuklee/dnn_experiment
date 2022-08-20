#!/usr/bin/python2
#-*- coding:utf-8 -*-

"""!
@file gen_vtkscene.py
@brief Generate virtual random scens at vtk_dataset with vtk.

It requires below prebuild dependencies.

```bash
    sudo apt-get install libvtk6-qt-dev

    # pip must be 20.0.1 : https://github.com/pypa/pip/issues/7620
    python2 -m pip install --user --force-reinstall pip==20.0.1
    pip2 install --user cython

    # Please see https://github.com/pypa/pip/issues/5599 for advice on fixing the underlying issue.
    # To avoid this problem you can invoke Python with '-m pip' instead of running pip directly. 
    python2 -m pip install deepdish==0.2.0 numexpr==2.6.2 tables==3.5.2
```

"""

import vtk

import numpy as np
from vtk.util import numpy_support
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vtk.util.numpy_support import vtk_to_numpy
import cv2

from os import path as osp
from os import makedirs
import glob
import argparse
import shutil

#import deepdish as dd
import pickle

from unet_ext import FindEdge, UnprojectPointscloud
from unet.util import colors, AddEdgeNoise, GetColoredLabel, remove_small_instance

class RBoxSource:
    def __init__(self, w, h, d, edge_round):
        self.corners = vtk.vtkPoints()
        self.corners.SetNumberOfPoints(8)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(self.corners)

        self.glyph_filter = vtk.vtkGlyph3D()
        self.glyph_filter.SetInputData(polydata)

        self.sphere_source = vtk.vtkSphereSource()
        resol = 100
        self.sphere_source.SetThetaResolution(resol)
        self.sphere_source.SetPhiResolution(resol)

        self.glyph_filter.SetSourceConnection(self.sphere_source.GetOutputPort())

        self.hull_filter = vtk.vtkHull()
        self.hull_filter.SetInputConnection(self.glyph_filter.GetOutputPort())
        self.hull_filter.AddCubeFacePlanes()
        self.hull_filter.AddRecursiveSpherePlanes(5);

        reader = vtk.vtkPNGReader()
        reader.SetFileName("/home/geo/Documents/texture0.png")
        self.texture = vtk.vtkTexture()
        self.texture.InterpolateOn()
        self.texture.SetInputConnection(reader.GetOutputPort())

        self.texturemap = vtk.vtkTextureMapToPlane()
        self.texturemap.SetInputConnection(self.hull_filter.GetOutputPort())

        self.SetEdgeRound(w, h, d, edge_round)


    def SetEdgeRound(self, w, h, d, edge_round):
        # TODO computing
        #r = edge_round / min(w,h)
        #hw = 0.5 - r
        #self.corners.SetPoint(0, hw, hw, hw)
        #self.corners.SetPoint(1, -hw, hw, hw)
        #self.corners.SetPoint(2, -hw, -hw, hw)
        #self.corners.SetPoint(3, hw, -hw, hw)
        #self.corners.SetPoint(4, hw, hw, -hw)
        #self.corners.SetPoint(5, -hw, hw, -hw)
        #self.corners.SetPoint(6, -hw, -hw, -hw)
        #self.corners.SetPoint(7, hw, -hw, -hw)
        #self.sphere_source.SetRadius(r)
        hw = w/2.-edge_round
        hh = h/2.-edge_round
        hd = d/2.-edge_round
        self.corners.SetPoint(0,  hw,  hh,  hd)
        self.corners.SetPoint(1, -hw,  hh,  hd)
        self.corners.SetPoint(2, -hw, -hh,  hd)
        self.corners.SetPoint(3,  hw, -hh,  hd)
        self.corners.SetPoint(4,  hw,  hh, -hd)
        self.corners.SetPoint(5, -hw,  hh, -hd)
        self.corners.SetPoint(6, -hw, -hh, -hd)
        self.corners.SetPoint(7,  hw, -hh, -hd)
        self.sphere_source.SetRadius(edge_round)

        self.texturemap.Modified()
        self.texturemap.Update()
        poly_data=self.texturemap.GetOutput()
        tcoords = poly_data.GetPointData().GetTCoords()

        k = 0
        for i in range(poly_data.GetNumberOfCells()):
            cell = poly_data.GetCell(i)
            xyz_points = cell.GetPoints()
            xyz_points = [xyz_points.GetPoint(j) for j in range(xyz_points.GetNumberOfPoints())]

            n_edge = 0
            for j, pt0 in enumerate(xyz_points):
                pt0 = np.array(pt0)
                pt1 = np.array( xyz_points[(j+1)%len(xyz_points)] )
                d = np.linalg.norm(pt1-pt0)
                if d > 0.5:
                    n_edge += 1
            if n_edge < 4:
                continue

            np_xyz = np.zeros((len(xyz_points), 3) )
            for j, pt0 in enumerate(xyz_points):
                np_xyz[j,:] = np.array(pt0)
            cp = 0.5 * ( np_xyz.max(axis=0) + np_xyz.min(axis=0) )
            max_axis = np.abs(cp).argmax()
            id_lists = cell.GetPointIds()
            id_lists = [id_lists.GetId(j) for j in range(id_lists.GetNumberOfIds())]
            n = np.zeros((3,))
            n[max_axis] = 1.
            for j, pt in enumerate(xyz_points):
                delta = pt - cp
                delta[0] /= 1.1*w
                delta[1] /= 1.1*h
                uv = np.delete(delta, max_axis) + np.array((0.5,0.5))
                vtk_id = id_lists[j]
                tcoords.SetTuple(vtk_id, uv.tolist())

    def GetOutputPort(self):
        return self.texturemap.GetOutputPort()

    def CreateTexturedActor(self):
        actor = vtk.vtkActor()
        # On/Off texture
        actor.SetTexture(self.texture)
        return actor


# https://stackoverflow.com/questions/17659362/get-depth-from-camera-for-each-pixel
class Scene:
    def __init__(self):
        self.ren = vtk.vtkRenderer()
        self.ren2 = vtk.vtkRenderer()
        self.ren2.SetActiveCamera(self.ren.GetActiveCamera())
        self.bg = vtk.vtkNamedColors().GetColor3d("Maroon")
        self.ren.SetBackground(self.bg)
        self.renwin = vtk.vtkRenderWindow()
        self.renwin.AddRenderer(self.ren)
        self.renwin.SetSize(1280, 960)
        #self.renwin.SetSize(640, 480)

        self.renwin2 = vtk.vtkRenderWindow()
        self.renwin2.AddRenderer(self.ren2)
        self.renwin2.SetSize(self.renwin.GetSize())

        #self.renwin.SetSize(640, 480)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
        self.iren.SetRenderWindow(self.renwin)
        self.iren.Initialize()

    def GetIntrinsic(self):
        cam = self.ren.GetActiveCamera()
        angle = cam.GetViewAngle()
        angle = np.deg2rad(angle)
        w, h = self.renwin.GetSize()
        fy = 0.5*h/np.tan(0.5*angle) # == fx
        tf = cam.GetModelTransformMatrix()
        fx = fy * tf.GetElement(0,0) / tf.GetElement(1,1)
        wc = cam.GetWindowCenter()
        assert wc[0]==0 and wc[1]==0
        cx, cy = 0.5*w, 0.5*h
        K = np.array( (fx, 0., cx,
                       0., fy, cy,
                       0., 0., 1.) ).reshape((3,3))
        D = np.zeros((4,)) # No distortion at vtk.
        return K, D

    def GetDepth(self):
        z_buffer = vtk.vtkFloatArray()
        width, height = self.renwin.GetSize()
        self.renwin.GetZbufferData( 0, 0, width - 1, height - 1, z_buffer)
        z_buffer_data_numpy = numpy_support.vtk_to_numpy(z_buffer)
        z_buffer_data_numpy = np.reshape(z_buffer_data_numpy, (-1, width))
        z_buffer_data_numpy = np.flipud(z_buffer_data_numpy)  # flipping along the first axis (y)

        # Belows are necessary to get real z_buffer_data_numpy
        camera = self.ren.GetActiveCamera()
        z_near, z_far = camera.GetClippingRange()
        numerator = 2.0 * z_near * z_far
        denominator = z_far + z_near - (2.0 * z_buffer_data_numpy - 1.0) * (z_far - z_near)
        depth = numerator / denominator
        non_depth_data_value = 0. # np.nan
        depth[z_buffer_data_numpy == 1.0] = non_depth_data_value
        return np.asarray(depth)

    def GetRgb(self):
        self.renwin.Render()
        #for i, actor in enumerate(self.box_actors):
        #    actor.GetProperty().LightingOff()
        vtk_win_im = vtk.vtkWindowToImageFilter()
        vtk_win_im.SetInput(self.renwin)
        vtk_win_im.Update()
        vtk_image = vtk_win_im.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()
        rgb = vtk_to_numpy(vtk_array).reshape(height, width, components)
        rgb = np.flipud(rgb)  # flipping along the first axis (y)
        return np.asarray(rgb)

    def GetMask(self,verbose=False):
        self.renwin2.Render()
        for target in ['vis','mask']:
            actors = self.ren2.GetActors()
            actors.InitTraversal()
            N = actors.GetNumberOfItems()
            if verbose:
                print(N)
            for i in range(N):
                actor = actors.GetNextProp()
                ins_id = self.ren2actor_to_id[actor]
                if target == 'vis':
                    color = [c/255. for c in colors[ins_id%len(colors)]]
                else:
                    color = [float(ins_id)/255.,0,0]
                actor.GetProperty().SetColor(color)
                actor.GetProperty().LightingOff()
                actor.SetTexture(None)

            vtk_win_im = vtk.vtkWindowToImageFilter()
            vtk_win_im.SetInput(self.renwin2)
            vtk_win_im.Update()
            vtk_image = vtk_win_im.GetOutput()
            width, height, _ = vtk_image.GetDimensions()
            vtk_array = vtk_image.GetPointData().GetScalars()
            components = vtk_array.GetNumberOfComponents()
            arr = vtk_to_numpy(vtk_array).reshape(height, width, components)
            arr = np.flipud(arr)  # flipping along the first axis (y)
            if target == 'vis':
                vis = np.asarray(arr)
            else:
                mask = np.asarray(arr)[:,:,0]
        return vis, mask

    def OnPress(self, event):
        if event.key == 'q':
            self.quit = True

    def GenerateSeparatedScene(self, dataset_name='vtk_dataset'):
        script_fn = osp.abspath(__file__)
        pkg_dir = str('/').join(script_fn.split('/')[:-2])
        dataset_path = osp.join(pkg_dir, dataset_name)
        verbose = True
        # TODO 1)
        if osp.exists(dataset_path):
            shutil.rmtree(dataset_path)
        makedirs(dataset_path)
        shutil.copyfile(__file__, osp.join(dataset_path,'gen_vtkscene.py') )

        usages = [('train',200), ('test',20)]
        for usage,n_image in usages:
            usage_path = osp.join(dataset_path, usage)
            makedirs(usage_path)
            vis_path = osp.join(usage_path, 'vis')
            makedirs(vis_path)
            #n_image = 10
            img_idx = 0
            while img_idx < n_image:
                print("%d/%d"%(img_idx,n_image) )
                for ren in [self.ren, self.ren2]:
                    actors = ren.GetActors()
                    actors.InitTraversal()
                    for i in range(actors.GetNumberOfItems()):
                        actor = actors.GetNextProp()
                        ren.RemoveActor(actor)
                self.MakeSeparatedBoxes()
                self.ren.ResetCameraClippingRange()
                self.ren2.ResetCameraClippingRange()

                vis, mask = self.GetMask()
                edge = FindEdge(mask.astype(np.uint8)) # Get instance edge
                #print('id_to_ren2actors.keys()=',self.id_to_ren2actors.keys())
                #print('unique(mask)=',np.unique(mask))
                #print('outliker=',outliers)
                mask[edge>0] = 0
                _, outliers = remove_small_instance(mask, min_width=100)
                for outlier_id in outliers:
                    #if not outlier_id in self.id_to_ren2actors:
                    #    continue
                    ren2actor = self.id_to_ren2actors[outlier_id]
                    ren_actor = self.ren2_to_ren_actors[ren2actor]
                    self.ren.RemoveActor(ren_actor)
                    self.ren2.RemoveActor(ren2actor)

                rgb = self.GetRgb()
                vis, mask0 = self.GetMask()
                mask = np.zeros_like(mask0)
                zeron_in_mask0 = 0 in mask0
                for ins1, ins0 in enumerate(np.unique(mask0)):
                    if zeron_in_mask0:
                        mask[mask0==ins0] = ins1
                    else:
                        mask[mask0==ins0] = ins1+1

                edge = FindEdge(mask.astype(np.uint8)) # Get instance edge
                mask[edge>0] = 0

                if mask.max() < 3:
                    continue

                black = np.logical_and(rgb[:,:,0]==0,rgb[:,:,1]==0,rgb[:,:,2]==0)
                if np.any(black):
                    print("Unexpected render failure")
                    continue

                org_depth = self.GetDepth()

                # Depth noise
                depth = org_depth + np.random.normal(0,.001,org_depth.size).reshape(org_depth.shape).astype(org_depth.dtype)
                depth[org_depth==0] = 0.
                #depth[np.logical_and(mask==0, edge==0)] = 0.
                K, D = self.GetIntrinsic()
                K = K.astype(np.float32)
                D = D.astype(np.float32)
                width, height = self.renwin.GetSize()
                info = {"K":K, "D":D, "width":width, "height":height}

                #box_face = np.logical_and(edge==0, mask >0)
                #box_face = box_face.astype(np.uint8)
                #retval, labels = cv2.connectedComponents(box_face)
                #dist, labels = cv2.distanceTransformWithLabels( (mask==0).astype(np.uint8),
                #        distanceType=cv2.DIST_L2, maskSize=5)
                #mask[dist > 7.] = 0
                depth[mask==0] = 0

                bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                cv2.imshow("mask",   GetColoredLabel(mask) )
                cv2.imshow("bgr",    bgr)
                cv2.imshow("depth", (depth>0).astype(np.uint8)*255 )

                #lights = self.ren.GetLights()
                #lights.InitTraversal()
                #for i in range(lights.GetNumberOfItems()):
                #    light = lights.GetNextItem()
                if cv2.waitKey(1) == ord('q'):
                    exit(1)

                cv2.imwrite(osp.join(vis_path,'%04d_bgr.png'%img_idx), bgr)
                cv2.imwrite(osp.join(vis_path,'%04d_markers.png'%img_idx), GetColoredLabel(mask) )

                rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                xyzrgb, ins_points0 = UnprojectPointscloud(rgb, depth, mask,
			        K, D, 0.02, 0.01)
                # Remove instance points which are reprojected on background due to numerical error of undistortion.
                xyzrgb = xyzrgb[ins_points0>0]
                ins_points = ins_points0[ins_points0>0]
                #print('unique ins = ', np.unique(mask), np.unique(ins_points) )
                #import pdb; pdb.set_trace()
                # TODO "edges", "markers"
                pick = { 'xyzrgb':xyzrgb, 'ins_points':ins_points,
                        'rgb':rgb, 'depth':depth,
                        'edge':edge, 'mask':mask,
                        'K':K, 'newK':K, 'D':D}

                fn = osp.join(usage_path, "%d.pick"%img_idx)
                with open(fn, 'wb') as f:
                    pickle.dump(pick, f, protocol=2)
                img_idx += 1

        return

    def MakeSeparatedBoxes(self):
        nr, nc = np.random.randint(low=2,high=7,size=2)
        camera = self.ren.GetActiveCamera()
        camera.SetViewAngle( 40. );
        camera.SetViewUp(0.,-1.,0.);
        w = np.random.uniform(.5, 1.5)
        h = np.random.uniform(.5, 1.5)
        d= .2
        #dx, dy = 0.05 , 0.05
        dx, dy = 0. , 0.
        #z = 3. # np.random.uniform(2.,6.)
        z = np.random.uniform(2., 4.)
        edge_round = .002
        margin = 0.01*edge_round
        cx = 0.5*nc*w #+ np.random.uniform(-0.5,0.5)
        cy = 0.5*nr*h #+ np.random.uniform(-0.5,0.5)
        camera.SetPosition(0,0,0.)
        camera.SetFocalPoint(0,0,z+1.)
        #camera.SetClippingRange(0.1, 99.)
        rbox_source = RBoxSource(w+margin,h+margin, d, edge_round)
        box_mapper = vtk.vtkPolyDataMapper()
        box_mapper.SetInputConnection(rbox_source.GetOutputPort())
        self.ren2_to_ren_actors = {}
        self.id_to_ren2actors = {}
        self.ren2actor_to_id = {}
        i = 1
        for r in range(nr):
            y = (h + dy )* float(r) - cy
            for c in range(nc):
                for ren in [self.ren, self.ren2]:
                    x = (w + dx) * float(c) - cx
                    actor = rbox_source.CreateTexturedActor()
                    actor.SetMapper(box_mapper)
                    actor.SetPosition(x, y, z)
                    actor.Modified()
                    ren.AddActor(actor)
                    if ren == self.ren:
                        actor0 = actor
                    else:
                        self.id_to_ren2actors[i] = actor
                        self.ren2actor_to_id[actor] = i
                        self.ren2_to_ren_actors[actor] = actor0
                        i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, nargs='?', help='The name of output dataset')
    args = parser.parse_args()
    if args.dataset_name==None:
        args.dataset_name = 'vtk_dataset'
    scene = Scene()
    scene.GenerateSeparatedScene(args.dataset_name)
