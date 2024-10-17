# trace generated using paraview version 5.13.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
data_420vtu = XMLUnstructuredGridReader(registrationName='data_420.vtu', FileName=['/home/innerbergerm@hhmi.org/Projects/janelia/embryo-harmonics/results/data_420.vtu'])

# Properties modified on data_420vtu
data_420vtu.PointArrayStatus = ['harmonic_013']
data_420vtu.TimeArray = 'None'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
data_420vtuDisplay = Show(data_420vtu, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
data_420vtuDisplay.Representation = 'Surface'

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=data_420vtu)

# Properties modified on contour1
contour1.Isosurfaces = [-0.0014258998800048756, -0.001109038437013861, -0.0007921769940228464, -0.00047531555103183173, -0.0001584541080408171, 0.00015840733495019752, 0.00047526877794121214, 0.000792130220932227, 0.0011089916639232414, 0.0014258531069142558, 0.0017427145499052707]

# show data in view
contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
contour1Display.Representation = 'Surface'

# hide data in view
Hide(data_420vtu, renderView1)

# show color bar/color legend
contour1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get color transfer function/color map for 'harmonic_013'
harmonic_013LUT = GetColorTransferFunction('harmonic_013')

# Rescale transfer function
harmonic_013LUT.RescaleTransferFunction(-0.001109038437013861, 0.0017427145499052707)

# get opacity transfer function/opacity map for 'harmonic_013'
harmonic_013PWF = GetOpacityTransferFunction('harmonic_013')

# Rescale transfer function
harmonic_013PWF.RescaleTransferFunction(-0.001109038437013861, 0.0017427145499052707)

# get 2D transfer function for 'harmonic_013'
harmonic_013TF2D = GetTransferFunction2D('harmonic_013')

# Properties modified on contour1Display
contour1Display.ComputePointNormals = 1

# Properties modified on contour1Display
contour1Display.FeatureAngle = 77.4

# Properties modified on contour1Display
contour1Display.FeatureAngle = 144.0

# Properties modified on contour1Display
contour1Display.FeatureAngle = 180.0

# Properties modified on contour1Display
contour1Display.FeatureAngle = 0.0

# Properties modified on contour1Display
contour1Display.FeatureAngle = 91.8

# Properties modified on contour1Display
contour1Display.ComputePointNormals = 0

# Properties modified on contour1Display
contour1Display.ComputePointNormals = 1

# Properties modified on contour1Display
contour1Display.ComputePointNormals = 0

renderView1.ResetActiveCameraToPositiveY()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.0491071455180645, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.1071428656578064, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.1830357164144516, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.2187500149011612, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.267857164144516, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.3392857313156128, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.3839285969734192, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.4330357313156128, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.4732142984867096, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.5446428656578064, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.6116071939468384, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.691964328289032, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.7276785969734192, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.7767857313156128, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8080357313156128, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.848214328289032, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8750000596046448, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.9017857313156128, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.9196429252624512, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.941964328289032, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.973214328289032, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.9910714626312256, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.00028424657648429275, 0.9866071939468384, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.00028424657648429275, 0.9821429252624512, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0002516551176086068, 0.8437500596046448, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0001620286057004705, 0.299107164144516, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.00014573287626262754, 0.0, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.00015388074098154902, 0.0, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0001620286057004705, 0.0, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.000170176470419392, 0.0, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.00017832433513831347, 0.0, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.00018647219985723495, 0.0, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.00019462006457615644, 0.0, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.00020276792929507792, 0.0, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.00022721152345184237, 0.0, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.00024350725288968533, 0.0, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0002598029968794435, 0.0, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0002679508470464498, 0.0, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013LUT
harmonic_013LUT.EnableOpacityMapping = 1

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.00028424657648429275, 0.008928571827709675, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.00028424657648429275, 0.0491071455180645, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.00028424657648429275, 0.0937500074505806, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.00028424657648429275, 0.1473214328289032, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0002760987263172865, 0.1696428656578064, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0002760987263172865, 0.1875000149011612, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0002679508470464498, 0.2142857313156128, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0002598029968794435, 0.2455357313156128, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0002598029968794435, 0.2857142984867096, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0002598029968794435, 0.3080357313156128, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0002598029968794435, 0.330357164144516, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0002598029968794435, 0.3392857313156128, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0002598029968794435, 0.3437500298023224, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0002598029968794435, 0.3482142984867096, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0002598029968794435, 0.3571428656578064, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 1.0, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.9821429252624512, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.9508929252624512, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.9151785969734192, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8973214626312256, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8883929252624512, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8839285969734192, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8616071939468384, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8392857313156128, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8258928656578064, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8080357313156128, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8035714626312256, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 1.0, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8035714626312256, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 0.9821429252624512, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8035714626312256, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 0.9508929252624512, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8035714626312256, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 0.9241071939468384, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8035714626312256, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 0.910714328289032, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8035714626312256, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 0.8928571939468384, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8035714626312256, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 0.8750000596046448, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8035714626312256, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 0.8526785969734192, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8035714626312256, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 0.8303571939468384, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8035714626312256, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 0.816964328289032, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8035714626312256, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 0.8125000596046448, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8035714626312256, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 0.8080357313156128, 0.5, 0.0]

# Properties modified on harmonic_013PWF
harmonic_013PWF.Points = [-0.001109038437013861, 0.8035714626312256, 0.5, 0.0, 0.0002598029968794435, 0.361607164144516, 0.5, 0.0, 0.0017427145499052704, 0.8035714626312256, 0.5, 0.0]

# set active source
SetActiveSource(data_420vtu)

# set active source
SetActiveSource(data_420vtu)

# show data in view
data_420vtuDisplay = Show(data_420vtu, renderView1, 'UnstructuredGridRepresentation')

# Properties modified on data_420vtuDisplay
data_420vtuDisplay.ComputePointNormals = 1

# Properties modified on data_420vtuDisplay
data_420vtuDisplay.Opacity = 0.1

# Hide orientation axes
renderView1.OrientationAxesVisibility = 0

# set active source
SetActiveSource(contour1)

# hide color bar/color legend
contour1Display.SetScalarBarVisibility(renderView1, False)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(846, 870)

# current camera placement for renderView1
renderView1.CameraPosition = [-462.13205668783905, -935.8912045049376, 260.11051485931677]
renderView1.CameraFocalPoint = [0.3403463380246022, -0.03789515783022056, 252.21155236126828]
renderView1.CameraViewUp = [0.008100730236406574, 0.004436872050006553, 0.999957345258311]
renderView1.CameraParallelScale = 270.1858280738312

# save screenshot
SaveScreenshot(filename='/home/innerbergerm@hhmi.org/Projects/janelia/embryo-harmonics/images/harmonic_013_iso.png', viewOrLayout=renderView1, location=16, ImageResolution=[846, 870],
    TransparentBackground=1)

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(846, 870)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [-462.13205668783905, -935.8912045049376, 260.11051485931677]
renderView1.CameraFocalPoint = [0.3403463380246022, -0.03789515783022056, 252.21155236126828]
renderView1.CameraViewUp = [0.008100730236406574, 0.004436872050006553, 0.999957345258311]
renderView1.CameraParallelScale = 270.1858280738312


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://www.paraview.org/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------