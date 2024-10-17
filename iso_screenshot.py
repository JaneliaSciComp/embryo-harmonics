# trace generated using paraview version 5.13.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

def produce_screenshot(harmonic_number):
    # Find harmonic name as stored in the data
    harmonic_name = f'harmonic_{harmonic_number:03d}'
    ResetSession()

    # Load data and select the array to visualize
    data = XMLUnstructuredGridReader(registrationName='data', FileName=['/home/innerbergerm@hhmi.org/Projects/janelia/embryo-harmonics/results/data_420.vtu'])
    data.PointArrayStatus = [harmonic_name]
    data.TimeArray = 'None'

    # Get active view
    render_view = GetActiveViewOrCreate('RenderView')
    render_view.OrientationAxesVisibility = 0

    # Show surface of the data, smooth it, and make it transparent
    data_display = Show(data, render_view, 'UnstructuredGridRepresentation')
    data_display.Representation = 'Surface'
    data_display.ComputePointNormals = 1
    data_display.Opacity = 0.1

    # Create a contour with some isosurfaces
    contour = Contour(registrationName='Contour', Input=data)
    contour.Isosurfaces = [-0.0014258998800048756, -0.001109038437013861, -0.0007921769940228464, -0.00047531555103183173, -0.0001584541080408171, 0.00015840733495019752, 0.00047526877794121214, 0.000792130220932227, 0.0011089916639232414, 0.0014258531069142558, 0.0017427145499052707]

    # Show contour, make it a surface, and hide the color bar
    contour_display = Show(contour, render_view, 'GeometryRepresentation')
    contour_display.Representation = 'Surface'
    contour_display.SetScalarBarVisibility(render_view, False)

    # Set active source
    SetActiveSource(contour)

    # Get color and opacity map for contour
    lookup_table = GetColorTransferFunction(harmonic_name)
    lookup_table.RescaleTransferFunction(-0.001109038437013861, 0.0017427145499052707)
    lookup_table.EnableOpacityMapping = 1

    opacity_lut = GetOpacityTransferFunction(harmonic_name)
    opacity_lut.RescaleTransferFunction(-0.001109038437013861, 0.0017427145499052707)
    opacity_lut.Points = [-0.001109038437013861, 0.8035714626312256, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 0.8035714626312256, 0.5, 0.0]

    # current camera placement for renderView1
    render_view.CameraPosition = [-462.13205668783905, -935.8912045049376, 260.11051485931677]
    render_view.CameraFocalPoint = [0.3403463380246022, -0.03789515783022056, 252.21155236126828]
    render_view.CameraViewUp = [0.008100730236406574, 0.004436872050006553, 0.999957345258311]
    render_view.CameraParallelScale = 270.1858280738312

    # save screenshot
    SaveScreenshot(filename=f'/home/innerbergerm@hhmi.org/Projects/janelia/embryo-harmonics/images/{harmonic_name}_iso_debug.png', viewOrLayout=render_view, location=16, ImageResolution=[846, 870],
                   TransparentBackground=1)


produce_screenshot(13)
produce_screenshot(14)