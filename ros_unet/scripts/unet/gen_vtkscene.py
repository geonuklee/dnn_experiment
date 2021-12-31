#!/usr/bin/python2
#-*- coding:utf-8 -*-

"""!
@file gen_vtkscene.py
@brief Generate virtual random scens at vtk_dataset with vtk.

It requires below prebuild dependencies.

```bash
    sudo apt-get install libvtk6-qt-dev

    # If latest installation of later version is possible, try that.
    pip2 install -U deepdish==0.2.0
    pip2 install -U tables numexpr==2.6.2
    pip3 install -U tables==3.6.1

```

"""

import vtk

import numpy as np
from vtk.util import numpy_support
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy
import cv2

from os import path as osp
from os import makedirs
import glob
import argparse
import shutil

import deepdish as dd

import unet_cpp_extension2 as cpp_ext
from util import colors


class RBoxSource:
    def __init__(self):
        self.corners = vtk.vtkPoints()
        self.corners.SetNumberOfPoints(8)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(self.corners)

        self.glyph_filter = vtk.vtkGlyph3D()
        self.glyph_filter.SetInputData(polydata)

        self.sphere_source = vtk.vtkSphereSource()
        resol = 20
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

        # TODO move below?
        self.SetRoundRatio(0.05)


    def SetRoundRatio(self, radius):
        hw = 0.5 - radius
        self.corners.SetPoint(0, hw, hw, hw)
        self.corners.SetPoint(1, -hw, hw, hw)
        self.corners.SetPoint(2, -hw, -hw, hw)
        self.corners.SetPoint(3, hw, -hw, hw)
        self.corners.SetPoint(4, hw, hw, -hw)
        self.corners.SetPoint(5, -hw, hw, -hw)
        self.corners.SetPoint(6, -hw, -hw, -hw)
        self.corners.SetPoint(7, hw, -hw, -hw)
        self.sphere_source.SetRadius(radius)

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
        actor.SetTexture(self.texture)
        actor.GetProperty().SetLineWidth(20)
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

    def PltShow(self,verbose):
        script_fn = osp.abspath(__file__)
        pkg_dir = str('/').join(script_fn.split('/')[:-3])
        dataset_path = osp.join(pkg_dir, 'vtk_dataset')

        usages = ['train', 'valid']
        numbers = [500, 20]
        if not verbose:
            if osp.exists(dataset_path):
                shutil.rmtree(dataset_path)
            makedirs(dataset_path)
            makedirs(osp.join(dataset_path, 'src') )
            for usage in usages:
                usagepath = osp.join(dataset_path,'src',usage)
                if not osp.exists(usagepath):
                    makedirs(usagepath)

        if verbose:
            fig, ax = plt.subplots()
            fig.canvas.mpl_connect('key_press_event', self.OnPress)
        self.quit = False
        for k, usage in enumerate(usages):

            for img_idx in range(numbers[k]):
                if self.quit:
                    break
                print("%d/%d"%(img_idx, numbers[k]) )

                if hasattr(self, 'box_actors'):
                    for actor in self.box_actors:
                        self.ren.RemoveActor(actor)
                self.box_actors = []

                if img_idx < 0.1*numbers[k]:
                    self.MakeAlignedStack()
                else:
                    self.MakeRandomStack()

                self.renwin.Render()

                rgb = self.GetRgb()
                vis, mask = self.GetMask()
                edge = cpp_ext.FindEdge(mask.astype(np.uint8)) # Get instance edge

                org_depth = self.GetDepth()
                org_depth[ np.sum(vis,axis=2) == 0] = 0.

                depth = org_depth.copy()
                dist = cv2.distanceTransform( (edge<1).astype(np.uint8)*255, cv2.DIST_L2,5)
                #depth, dist, faint = AddRounding(org_depth, edge) # TODO Remove it.

                noise = np.random.normal(0.,0.0002,depth.shape)
                depth[depth>0] += noise[depth>0]

                lap5 =cv2.Laplacian(depth, cv2.CV_32FC1, ksize=5)
                dst_label = np.zeros((org_depth.shape[0], org_depth.shape[1],3), np.uint8)
                dst_label[org_depth>0,2] = 255
                dst_label[edge>0,:] = 255

                minirgb  = cv2.resize(rgb, (200,200) )
                minirgb = cv2.cvtColor(minirgb,cv2.COLOR_RGB2BGR)
                dst = np.zeros((dst_label.shape[0],
                                dst_label.shape[1]+minirgb.shape[1],3), np.uint8)
                for i in range(3):
                    dst[minirgb.shape[0]:,dst_label.shape[1]:,i] = 100
                dst[:,:dst_label.shape[1],:] = dst_label
                dst[:minirgb.shape[0],dst_label.shape[1]:,:] = minirgb

                K, D = self.GetIntrinsic()
                width, height = self.renwin.GetSize()
                info = {"K":K, "D":D, "width":width, "height":height}

                fn_form = osp.join(dataset_path, 'src', usage,"%d_%s.%s")
                if not verbose:
                    np.save(fn_form%(img_idx,"depth","npy"),depth)
                    np.save(fn_form%(img_idx,"rgb","npy"),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                    cv2.imwrite(fn_form%(img_idx,"gt","png"), dst)

                    # https://stackoverflow.com/questions/18071075/saving-dictionaries-to-file-numpy-and-python-2-3-friendly
                    dd.io.save(fn_form%(img_idx,"info","h5"), info, compression=('blosc', 9))
                    continue

                cv2.imshow("dst", dst)

                #cv2.imshow("faint", faint)
                r = cv2.normalize(dist,None,255,0,cv2.NORM_MINMAX,cv2.CV_8UC1)
                cv2.imshow("dist", r)

                r = cv2.normalize(depth,None,255,0,cv2.NORM_MINMAX,cv2.CV_8UC1)
                cv2.imshow("depth", r)

                cv2.imshow("lap5", 255*(lap5 < -0.1).astype(np.uint8))
                #cv2.imshow("edge", 255*edge)

                plt.subplot(131).title.set_text('depth map')
                plt.imshow(depth)

                plt.subplot(132).title.set_text('org image')
                plt.imshow(rgb)

                plt.subplot(133).title.set_text('vis mask')
                plt.imshow(vis)

                plt.suptitle('Move cursor on iamge and see value')
                plt.draw()
                plt.waitforbuttonpress(timeout=0.01)

                c=cv2.waitKey()
                self.quit=c==ord('q')

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
        margin = 0.01 # To remove gap.

        camera.SetFocalPoint(0.5*nc*w,0.5*nr*h,z)
        cx = 0.5*nc*w + np.random.uniform(-0.1,0.1)
        cy = 0.5*nr*h + np.random.uniform(-0.1,0.1)
        camera.SetPosition(cx, cy, 0.)

        rbox_source = RBoxSource()
        box_mapper = vtk.vtkPolyDataMapper()
        box_mapper.SetInputConnection(rbox_source.GetOutputPort())

        for r in range(nr):
            y = h*float(r)
            for c in range(nc):
                x = w*float(c)
                actor = rbox_source.CreateTexturedActor()
                actor.SetMapper(box_mapper)
                actor.SetPosition(x, y, z)
                actor.SetScale(w+margin, h+margin, d)
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

        rbox_source = RBoxSource()
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
                actor.SetScale(w, h, d)
                self.ren.AddActor(actor)
                self.box_actors.append(actor)
                th0, th1 = np.random.uniform(-40.,40.,2)
                actor.RotateX(th0)
                actor.RotateY(th1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "--verbose", "-v", action="count", help="verbose")
    args = parser.parse_args()
    scene = Scene()
    scene.PltShow(args.verbose)
