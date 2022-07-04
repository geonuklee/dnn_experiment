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
from unet.util import colors, AddEdgeNoise, GetColoredLabel

box_round = .005

class RBoxSource:
    def __init__(self, w, h, d):
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
        reader.SetFileName("/home/geo/Documents/texture.png")

        self.texture = vtk.vtkTexture()
        self.texture.InterpolateOn()
        self.texture.SetInputConnection(reader.GetOutputPort())

        self.texturemap = vtk.vtkTextureMapToPlane()
        self.texturemap.SetInputConnection(self.hull_filter.GetOutputPort())

        self.SetRoundRatio(w, h, d, box_round)
        #self.SetRoundRatio(w, h, d, 0.05) # Train sucess


    def SetRoundRatio(self, w, h, d, round_ratio):
        r = round_ratio/w
        hw = 0.5 - r
        self.corners.SetPoint(0, hw, hw, hw)
        self.corners.SetPoint(1, -hw, hw, hw)
        self.corners.SetPoint(2, -hw, -hw, hw)
        self.corners.SetPoint(3, hw, -hw, hw)
        self.corners.SetPoint(4, hw, hw, -hw)
        self.corners.SetPoint(5, -hw, hw, -hw)
        self.corners.SetPoint(6, -hw, -hw, -hw)
        self.corners.SetPoint(7, hw, -hw, -hw)
        self.sphere_source.SetRadius(r)

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
                uv = np.delete(delta, max_axis) + np.array((0.5,0.5))
                vtk_id = id_lists[j]
                tcoords.SetTuple(vtk_id, uv.tolist())
        #self.poly_data = poly_data

    def GetOutputPort(self):
        return self.texturemap.GetOutputPort()

    def CreateTexturedActor(self):
        actor = vtk.vtkActor()
        #actor.SetTexture(self.texture)
        return actor


# https://stackoverflow.com/questions/17659362/get-depth-from-camera-for-each-pixel
class Scene:
    def __init__(self):
        self.ren = vtk.vtkRenderer()
        self.renwin = vtk.vtkRenderWindow()
        self.renwin.AddRenderer(self.ren)
        self.renwin.SetSize(1280, 960)
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

    def GetMask(self):
        for target in ['vis','mask']:
            for i, actor in enumerate(self.box_actors):
                if target == 'vis':
                    color = [c/255. for c in colors[i%len(colors)]]
                else:
                    color = [float(i+1)/255.,0,0]
                actor.GetProperty().SetColor(color)
                actor.GetProperty().LightingOff()
                actor.SetTexture(None)

            vtk_win_im = vtk.vtkWindowToImageFilter()
            vtk_win_im.SetInput(self.renwin)
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

        usages = [('train',200), ('test',10)]
        for usage,n_image in usages:
            usage_path = osp.join(dataset_path, usage)
            makedirs(usage_path)
            #n_image = 10
            for img_idx in range(n_image):
                print("%d/%d"%(img_idx,n_image) )
                if hasattr(self, 'box_actors'):
                    for actor in self.box_actors:
                        self.ren.RemoveActor(actor)
                self.box_actors = []
                self.MakeSeparatedBoxes()
                self.ren.ResetCameraClippingRange()
                self.renwin.Render()
                rgb = self.GetRgb()
                vis, mask = self.GetMask()
                edge = FindEdge(mask.astype(np.uint8)) # Get instance edge
                mask[edge>0] = 0
                org_depth = self.GetDepth()
                org_depth[ np.sum(vis,axis=2) == 0] = 0.
                #depth = org_depth.copy()
                depth = org_depth + np.random.normal(0,.001,org_depth.size).reshape(org_depth.shape).astype(org_depth.dtype)
                depth[org_depth==0] = 0.

                dst_label = np.zeros((org_depth.shape[0], org_depth.shape[1],3), np.uint8)
                dst_label[org_depth>0,2] = 255
                #dst_label[edge>0,:] = 255

                minirgb  = cv2.resize(rgb, (200,200) )
                minirgb = cv2.cvtColor(minirgb,cv2.COLOR_RGB2BGR)
                dst = np.zeros((dst_label.shape[0],
                                dst_label.shape[1]+minirgb.shape[1],3), np.uint8)
                for i in range(3):
                    dst[minirgb.shape[0]:,dst_label.shape[1]:,i] = 100
                dst[:,:dst_label.shape[1],:] = dst_label
                dst[:minirgb.shape[0],dst_label.shape[1]:,:] = minirgb
                K, D = self.GetIntrinsic()
                K = K.astype(np.float32)
                D = D.astype(np.float32)
                width, height = self.renwin.GetSize()
                info = {"K":K, "D":D, "width":width, "height":height}

                #box_face = np.logical_and(edge==0, mask >0)
                #box_face = box_face.astype(np.uint8)
                #retval, labels = cv2.connectedComponents(box_face)
                dist, labels = cv2.distanceTransformWithLabels( (mask==0).astype(np.uint8),
                        distanceType=cv2.DIST_L2, maskSize=5)
                labels[dist > 7.] = 0

                # I have no idea reason for need this. but without it, cpp receive wrong rgb.
                bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                cv2.imshow("bgr", bgr)
                if cv2.waitKey(1) == ord('q'):
                    exit(1)
                rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                xyzrgb, ins_points = UnprojectPointscloud(rgb, depth, labels,
                        K, D, 0.02, 0.01)
                #print('unique ins = ', np.unique(ins_points) )
                pick = { 'xyzrgb':xyzrgb, 'ins_points':ins_points, 'rgb':rgb, 'K':K, 'D':D}

                fn = osp.join(usage_path, "%d.pick"%img_idx)
                with open(fn, 'wb') as f:
                    pickle.dump(pick, f, protocol=2)

        return

    def MakeSeparatedBoxes(self):
        nr, nc = np.random.randint(low=2,high=7,size=2)
        camera = self.ren.GetActiveCamera()
        camera.SetViewAngle( 40. );
        camera.SetViewUp(0.,-1.,0.);
        w = np.random.uniform(.2,.8)
        h = np.random.uniform(.2,.8)
        #dx = np.random.uniform(-.2,.2)
        #dy = np.random.uniform(-.2,.2)
        dx, dy = .05, .05
        d= .1
        #z = 3. # np.random.uniform(2.,6.)
        z = np.random.uniform(2.,6.)
        margin = 0.05*box_round
        cx = 0.5*nc*w #+ np.random.uniform(-0.5,0.5)
        cy = 0.5*nr*h #+ np.random.uniform(-0.5,0.5)
        camera.SetPosition(0,0,0.)
        camera.SetFocalPoint(0,0,z+1.)
        #camera.SetClippingRange(0.1, 99.)
        rbox_source = RBoxSource(w+margin,h+margin, d)
        box_mapper = vtk.vtkPolyDataMapper()
        box_mapper.SetInputConnection(rbox_source.GetOutputPort())
        for r in range(nr):
            y = (h + dy )* float(r) - cy
            for c in range(nc):
                x = (w + dx) * float(c) - cx
                actor = rbox_source.CreateTexturedActor()
                actor.SetMapper(box_mapper)
                actor.SetPosition(x, y, z)
                actor.SetScale(w,h,d)
                actor.Modified()
                self.ren.AddActor(actor)
                self.box_actors.append(actor)

    def MakeAlignedStack(self):
        camera = self.ren.GetActiveCamera()
        camera.SetViewAngle( 40. );
        dx = np.random.uniform(-0.3,0.3)
        camera.SetViewUp(dx,-1.,0.);

        nr, nc = 10,20
        w = np.random.uniform(0.3,0.8)
        h = np.random.uniform(0.5,2.)*w
        d=1.
        z = np.random.uniform(2.,6.)
        margin = 0. # To remove gap.

        camera.SetFocalPoint(0.5*nc*w,0.5*nr*h,z)
        cx = 0.5*nc*w + np.random.uniform(-0.1,0.1)
        cy = 0.5*nr*h + np.random.uniform(-0.1,0.1)
        camera.SetPosition(cx, cy, 0.)

        rbox_source = RBoxSource(w+margin,h+margin, d)
        box_mapper = vtk.vtkPolyDataMapper()
        box_mapper.SetInputConnection(rbox_source.GetOutputPort())

        for r in range(nr):
            y = h*float(r)
            for c in range(nc):
                x = w*float(c)
                actor = rbox_source.CreateTexturedActor()
                actor.SetMapper(box_mapper)
                actor.SetPosition(x, y, z)
                self.ren.AddActor(actor)
                self.box_actors.append(actor)

    def MakeRandomStack(self):
        camera = self.ren.GetActiveCamera()
        camera.SetViewAngle( 40. );
        camera.SetViewUp(0.,-1.,0.);

        nr, nc = 10, 20
        w0 = np.random.uniform(0.3,0.8)
        h0 = np.random.uniform(0.5,2.)*w0
        d=1.
        z0 = 5.
        cx, cy = w0*float(nc)/2., h0*float(nr)/2.
        tx = np.random.uniform(cx-0.1,cx+0.1)
        ty = np.random.uniform(cy-0.5, cy+0.5)
        camera.SetFocalPoint(cx, cy, z0);
        camera.SetPosition(tx, ty, 0.);

        rbox_source = RBoxSource(w0,h0,d)
        box_mapper = vtk.vtkPolyDataMapper()
        box_mapper.SetInputConnection(rbox_source.GetOutputPort())

        for r in range(nr):
            y0 = 1.2* h0*float(r)
            for c in range(nc):
                if len(self.box_actors) >= 253:
                    break
                x0 = 1.2*  w0*float(c)
                x = np.random.uniform(x0, x0+0.5)
                y = np.random.uniform(y0, y0+0.5)
                z = np.random.uniform(z0,z0+1.)
                h = np.random.uniform(0.9*h0, 1.2*h0)
                w = np.random.uniform(0.9*w0, 1.2*w0)

                actor = rbox_source.CreateTexturedActor()
                actor.SetMapper(box_mapper)
                actor.SetPosition(x, y, z)
                self.ren.AddActor(actor)
                self.box_actors.append(actor)
                th0, th1 = np.random.uniform(-40.,40.,2)
                actor.RotateX(th0)
                actor.RotateY(th1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, nargs='?', help='The name of output dataset')
    args = parser.parse_args()
    if args.dataset_name==None:
        args.dataset_name = 'vtk_dataset'
    scene = Scene()
    scene.GenerateSeparatedScene(args.dataset_name)
