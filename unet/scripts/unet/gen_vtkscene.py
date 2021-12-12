#!/usr/bin/python2
#-*- coding:utf-8 -*-

# sudo apt-get install libvtk6-qt-dev
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

# TODO Replace it to pybind11 cpp extension
import unet_cpp_extension as cpp_ext

colors = (
  (0,255,0),
  (0,180,0),
  (0,100,0),
  (255,0,255),
  (100,0,255),
  (255,0,100),
  (100,0,100),
  (0,0,255),
  (0,0,180),
  (0,0,100),
  (255,255,0),
  (100,255,0),
  (255,100,0),
  (100,100,0),
  (255,0,0),
  (180,0,0),
  (100,0,0),
  (0,255,255),
  (0,100,255),
  (0,255,100),
  (0,100,100)
)

def AddRounding(org_depth, edge):
    depth = org_depth.copy()
    dist = cv2.distanceTransform( (edge<1).astype(np.uint8)*255, cv2.DIST_L2,5)

    dmax = 2. # 3pixel
    boundary = np.logical_and(dist < dmax, depth > 0.)
    boundary = np.logical_and(boundary, np.random.uniform(0.,1.,edge.shape) < 0.02)
    faint = 255 * boundary.astype(np.uint8)

    r = 0.002 # rounding depth 1cm
    tmp = r**2-np.power(r/dmax*dist[boundary]-r,2) #print np.min(tmp)
    tmp[tmp<0] = 0.
    rounding = r - np.sqrt(tmp)
    #print "max rounding = ", np.max(rounding)
    depth[boundary] += rounding
    return depth, dist, faint


# https://stackoverflow.com/questions/17659362/get-depth-from-camera-for-each-pixel
class Scene:
    def __init__(self):
        self.ren = vtk.vtkRenderer()
        self.renwin = vtk.vtkRenderWindow()
        self.renwin.AddRenderer(self.ren)
        self.renwin.SetSize(500,400)
        vtk_render_window_interactor = vtk.vtkRenderWindowInteractor()
        vtk_render_window_interactor.SetRenderWindow(self.renwin)
        vtk_render_window_interactor.Initialize()

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
        pkg_dir = str('/').join(script_fn.split('/')[:-2])
        dataset_path = osp.join(pkg_dir, 'edge_dataset')

        usages = ['train', 'valid', 'test']
        numbers = [100, 10,10]
        if not verbose:
            if osp.exists(dataset_path):
                shutil.rmtree(dataset_path)
            if True:
                makedirs(dataset_path)
                for usage in usages:
                    usagepath = osp.join(dataset_path,usage)
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

                depth, dist, faint = AddRounding(org_depth, edge)

                noise = np.random.normal(0.,0.0002,depth.shape)
                depth[depth>0] += noise[depth>0]
                depth[depth==0] = np.random.uniform(0.,0.1,depth.shape)[depth==0]

                lap =cv2.Laplacian(depth, cv2.CV_32FC1, ksize=5)

                if not verbose:
                    fn_form = osp.join(dataset_path, usage,'%d_'%img_idx+'%s')
                    np.save(fn_form%'edge',edge)
                    np.save(fn_form%'lap',edge)
                    np.save(fn_form%'depth',depth)
                    np.save(fn_form%'rgb',rgb)
                    #cv2.imwrite(fn_form%'rgb'+'.png', rgb)
                if not verbose:
                    continue

                cv2.imshow("faint", faint)
                r = cv2.normalize(dist,None,255,0,cv2.NORM_MINMAX,cv2.CV_8UC1)
                cv2.imshow("dist", r)

                r = cv2.normalize(depth,None,255,0,cv2.NORM_MINMAX,cv2.CV_8UC1)
                cv2.imshow("depth", r)

                cv2.imshow("lap", 255*(lap < -1.).astype(np.uint8))
                cv2.imshow("edge", 255*edge)

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

        camera.SetFocalPoint(0.5*nc*w,0.5*nr*h,z)
        cx = 0.5*nc*w + np.random.uniform(-0.1,0.1)
        cy = 0.5*nr*h + np.random.uniform(-0.1,0.1)
        camera.SetPosition(cx, cy, 0.)

        source = vtk.vtkCubeSource()
        source.SetXLength(1.)
        source.SetYLength(1.)
        source.SetZLength(1.)

        box_mapper = vtk.vtkPolyDataMapper()
        box_mapper.SetInputConnection(source.GetOutputPort())

        for r in range(nr):
            y = h*float(r)
            for c in range(nc):
                x = w*float(c)
                actor = vtk.vtkActor()
                actor.SetMapper(box_mapper)
                actor.SetPosition(x, y, z)
                actor.SetScale(w, h, d)
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


        source = vtk.vtkCubeSource()
        source.SetXLength(1.)
        source.SetYLength(1.)
        source.SetZLength(1.)
        box_mapper = vtk.vtkPolyDataMapper()
        box_mapper.SetInputConnection(source.GetOutputPort())


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

                actor = vtk.vtkActor()
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
