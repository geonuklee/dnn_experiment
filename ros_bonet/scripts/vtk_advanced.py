#!/usr/bin/python
#-*- coding:utf-8 -*-

import vtk
#assert( vtk.vtkVersion.GetVTKMajorVersion() >= 8)

class RBoxSource:
    def __init__(self):
        self.corners = vtk.vtkPoints()
        self.corners.SetNumberOfPoints(8)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(self.corners)

        self.glyph_filter = vtk.vtkGlyph3D()
        self.glyph_filter.SetInputData(polydata)

        self.sphere_source = vtk.vtkSphereSource()
        self.sphere_source.SetThetaResolution(30)
        self.sphere_source.SetPhiResolution(30)

        self.glyph_filter.SetSourceConnection(self.sphere_source.GetOutputPort())

        self.hull_filter = vtk.vtkHull()
        self.hull_filter.SetInputConnection(self.glyph_filter.GetOutputPort())
        self.hull_filter.AddCubeFacePlanes()
        self.hull_filter.AddRecursiveSpherePlanes(5);

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

    def GetOutputPort(self):
        #return self.glyph_filter.GetOutputPort()
        return self.hull_filter.GetOutputPort()


class Scene:
    def __init__(self):
        self.ren = vtk.vtkRenderer()
        self.renwin = vtk.vtkRenderWindow()
        self.renwin.AddRenderer(self.ren)
        self.renwin.SetSize(1280,960)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renwin)
        self.iren.Initialize()

    def draw(self):
        reader = vtk.vtkPNGReader()
        reader.SetFileName("/home/geo/Documents/texture.png")
        reader.Update()
        texture = vtk.vtkTexture()
        texture.InterpolateOn()

        source = RBoxSource()
        source.SetRoundRatio(0.05)
        box_mapper = vtk.vtkPolyDataMapper()


        #Map texture coordinates
        texturemap = vtk.vtkTextureMapToPlane()
        texturemap.SetInputConnection(source.GetOutputPort())
        box_mapper.SetInputConnection(texturemap.GetOutputPort())
        texture.SetInputConnection(reader.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(box_mapper)
        actor.SetPosition(0.,0.,0.)
        actor.SetScale(0.7, 1., 0.7)
        actor.GetProperty().SetColor(1.,1.,1.)
        actor.SetTexture(texture)
        actor.Modified()

        self.ren.AddActor(actor)

        self.renwin.Render()
        scene.iren.Start()


if __name__ == '__main__':
    scene = Scene()
    scene.draw()
