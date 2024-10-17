# trace generated using paraview version 5.13.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 13

#### import the simple module from the paraview
import os
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


def produce_screenshot(harmonic_number):
    # Find harmonic name as stored in the data
    harmonic_name = f'harmonic_{harmonic_number:03d}'
    Disconnect()
    Connect()

    # Load data and select the array to visualize
    data = XMLUnstructuredGridReader(registrationName='data_420.vtu', FileName=['/home/innerbergerm@hhmi.org/Projects/janelia/embryo-harmonics/results/data_420.vtu'])

    # Properties modified on data
    data.PointArrayStatus = [harmonic_name]
    data.TimeArray = 'None'

    # get active view
    render_view = GetActiveViewOrCreate('RenderView')

    # show data in view
    data_display = Show(data, render_view, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    data_display.Representation = 'Surface'

    # reset view to fit data
    render_view.ResetCamera(False, 0.9)

    # update the view to ensure updated data information
    render_view.Update()

    # set scalar coloring
    ColorBy(data_display, ('POINTS', harmonic_name))

    # rescale color and/or opacity maps used to include current data range
    data_display.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    data_display.SetScalarBarVisibility(render_view, True)

    # get color transfer function/color map for 'harmonic_013'
    color_lut = GetColorTransferFunction(harmonic_name)

    render_view.ResetActiveCameraToPositiveY()

    # reset view to fit data
    render_view.ResetCamera(False, 0.9)

    # Properties modified on data_420vtuDisplay
    data_display.ComputePointNormals = 1

    # Properties modified on harmonic_013LUT
    color_lut.NumberOfTableValues = 21

    # create a new 'Slice'
    slice_x = Slice(registrationName='Slice1', Input=data)

    # show data in view
    slice_x_display = Show(slice_x, render_view, 'GeometryRepresentation')

    # trace defaults for the display properties.
    slice_x_display.Representation = 'Surface'

    # hide data in view
    Hide(data, render_view)

    # show color bar/color legend
    slice_x_display.SetScalarBarVisibility(render_view, True)

    # update the view to ensure updated data information
    render_view.Update()

    # set active source
    SetActiveSource(data)

    # toggle interactive widget visibility (only when running from the GUI)
    HideInteractiveWidgets(proxy=slice_x.SliceType)

    # create a new 'Slice'
    slice_y = Slice(registrationName='Slice2', Input=data)

    # Properties modified on slice2.SliceType
    slice_y.SliceType.Normal = [0.0, 1.0, 0.0]

    # show data in view
    slice_y_display = Show(slice_y, render_view, 'GeometryRepresentation')

    # trace defaults for the display properties.
    slice_y_display.Representation = 'Surface'

    # hide data in view
    Hide(data, render_view)

    # show color bar/color legend
    slice_y_display.SetScalarBarVisibility(render_view, True)

    # update the view to ensure updated data information
    render_view.Update()

    # set active source
    SetActiveSource(slice_x)

    # toggle interactive widget visibility (only when running from the GUI)
    HideInteractiveWidgets(proxy=slice_y.SliceType)

    # toggle interactive widget visibility (only when running from the GUI)
    ShowInteractiveWidgets(proxy=slice_x.SliceType)

    # create a new 'Transform'
    transform1 = Transform(registrationName='Transform1', Input=slice_x)

    # Properties modified on transform1.Transform
    transform1.Transform.Translate = [220.0, 0.0, 0.0]

    # show data in view
    transform1Display = Show(transform1, render_view, 'GeometryRepresentation')

    # trace defaults for the display properties.
    transform1Display.Representation = 'Surface'

    # hide data in view
    Hide(slice_x, render_view)

    # show color bar/color legend
    transform1Display.SetScalarBarVisibility(render_view, True)

    # update the view to ensure updated data information
    render_view.Update()

    # set active source
    SetActiveSource(slice_y)

    # toggle interactive widget visibility (only when running from the GUI)
    HideInteractiveWidgets(proxy=transform1.Transform)

    # toggle interactive widget visibility (only when running from the GUI)
    ShowInteractiveWidgets(proxy=slice_y.SliceType)

    # create a new 'Transform'
    transform2 = Transform(registrationName='Transform2', Input=slice_y)

    # Properties modified on transform2.Transform
    transform2.Transform.Translate = [0.0, 220.0, 0.0]

    # show data in view
    transform2Display = Show(transform2, render_view, 'GeometryRepresentation')

    # trace defaults for the display properties.
    transform2Display.Representation = 'Surface'

    # hide data in view
    Hide(slice_y, render_view)

    # show color bar/color legend
    transform2Display.SetScalarBarVisibility(render_view, True)

    # update the view to ensure updated data information
    render_view.Update()

    # set active source
    SetActiveSource(data)

    # toggle interactive widget visibility (only when running from the GUI)
    HideInteractiveWidgets(proxy=transform2.Transform)

    # show data in view
    data_display = Show(data, render_view, 'UnstructuredGridRepresentation')

    # show color bar/color legend
    data_display.SetScalarBarVisibility(render_view, True)

    # hide color bar/color legend
    data_display.SetScalarBarVisibility(render_view, False)

    # Properties modified on renderView1
    render_view.OrientationAxesVisibility = 0

    # current camera placement for renderView1
    render_view.CameraPosition = [-750.0, -750.0, 237.0]
    render_view.CameraFocalPoint = [0.0, 0.0, 250]
    render_view.CameraViewUp = [0.0, 0.0, 1.0]
    render_view.CameraParallelScale = 270.0

    # save screenshot
    os.makedirs('/home/innerbergerm@hhmi.org/Projects/janelia/embryo-harmonics/images/projection_420', exist_ok=True)
    SaveScreenshot(filename=f'/home/innerbergerm@hhmi.org/Projects/janelia/embryo-harmonics/images/projection_420/{harmonic_name}_projection.png', viewOrLayout=render_view, location=16, ImageResolution=[1016, 870],
        TransparentBackground=1)


for i in range(100):
    produce_screenshot(i)
