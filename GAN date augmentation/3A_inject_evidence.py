from procedures.attack_pipeline import *

print('Removing Evidence...')

# Init pipeline
injector = scan_manipulator()

# Load target scan (provide path to dcm or mhd file)
path_to_target_scan = 'F:\\LUNA16_Dataset\\1.3.6.1.4.1.14519.5.2.1.6279.6001.100332161840553388986847034053.mhd'   # 设置要篡改的文件位置
injector.load_target_scan(path_to_target_scan)

# Inject at two locations (this version does not implement auto candidate location selection)
vox_coord1 = np.array([300,136,352]) #z, y , x (x-y should be flipped if the coordinates were obtained from an image viewer such as RadiAnt)
vox_coord2 = np.array([300,135,351])
vox_coord3 = np.array([300,134,350])
vox_coord4 = np.array([300,135,352])
injector.tamper(vox_coord1, action='inject', isVox=True)  # can supply realworld coord too
injector.tamper(vox_coord2, action='inject', isVox=True)
injector.tamper(vox_coord3, action='inject', isVox=True)
injector.tamper(vox_coord4, action='inject', isVox=True)

# Save scan
path_to_save_scan = 'data/tampered_scans'
injector.save_tampered_scan(path_to_save_scan, output_type='dicom')  # output can be dicom or numpy
