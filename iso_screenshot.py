# trace generated using paraview version 5.13.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
data = XMLUnstructuredGridReader(registrationName='data', FileName=['/home/innerbergerm@hhmi.org/Projects/janelia/embryo-harmonics/results/data_420.vtu'])

# Properties modified on data_420vtu
data.PointArrayStatus = ['harmonic_013']
data.TimeArray = 'None'

# get active view
render_view = GetActiveViewOrCreate('RenderView')

# show data in view
data_display = Show(data, render_view, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
data_display.Representation = 'Surface'

# show data in view
data_display = Show(data, render_view, 'UnstructuredGridRepresentation')

# Properties modified on data_420vtuDisplay
data_display.ComputePointNormals = 1

# Properties modified on data_420vtuDisplay
data_display.Opacity = 0.1

# create a new 'Contour'
contour = Contour(registrationName='Contour1', Input=data)

# Properties modified on contour1
contour.Isosurfaces = [-0.0014258998800048756, -0.001109038437013861, -0.0007921769940228464, -0.00047531555103183173, -0.0001584541080408171, 0.00015840733495019752, 0.00047526877794121214, 0.000792130220932227, 0.0011089916639232414, 0.0014258531069142558, 0.0017427145499052707]

# show data in view
contour_display = Show(contour, render_view, 'GeometryRepresentation')

# trace defaults for the display properties.
contour_display.Representation = 'Surface'

# hide color bar/color legend
contour_display.SetScalarBarVisibility(render_view, False)

# hide data in view
Hide(data, render_view)

# get color transfer function/color map for 'harmonic_013'
lookup_table = GetColorTransferFunction('harmonic_013')

# Rescale transfer function
lookup_table.RescaleTransferFunction(-0.001109038437013861, 0.0017427145499052707)

# get opacity transfer function/opacity map for 'harmonic_013'
opacity_lut = GetOpacityTransferFunction('harmonic_013')

# Rescale transfer function
opacity_lut.RescaleTransferFunction(-0.001109038437013861, 0.0017427145499052707)

# Properties modified on harmonic_013LUT
lookup_table.EnableOpacityMapping = 1

# Properties modified on harmonic_013PWF
opacity_lut.Points = [-0.001109038437013861, 0.8035714626312256, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 0.8035714626312256, 0.5, 0.0]

# set active source
SetActiveSource(data)

# set active source
SetActiveSource(data)

# Hide orientation axes
render_view.OrientationAxesVisibility = 0

# set active source
SetActiveSource(contour)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(846, 870)

# current camera placement for renderView1
render_view.CameraPosition = [-462.13205668783905, -935.8912045049376, 260.11051485931677]
render_view.CameraFocalPoint = [0.3403463380246022, -0.03789515783022056, 252.21155236126828]
render_view.CameraViewUp = [0.008100730236406574, 0.004436872050006553, 0.999957345258311]
render_view.CameraParallelScale = 270.1858280738312

# save screenshot
SaveScreenshot(filename='/home/innerbergerm@hhmi.org/Projects/janelia/embryo-harmonics/images/harmonic_013_iso_debug.png', viewOrLayout=render_view, location=16, ImageResolution=[846, 870],
               TransparentBackground=1)

ResetSession()