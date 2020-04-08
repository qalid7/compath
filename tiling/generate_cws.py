import CWS_generator as cws
import sys
import os
import glob


filename = sys.argv[1]
foldername = os.getcwd()
output_dir= os.path.join(os.getcwd(), 'cws')
print(filename)
_, filetype = os.path.splitext(filename)
if not filetype == '.czi':
    cws_obj = cws.CWSGENERATOR(output_dir=output_dir, file_name=filename, input_dir=foldername)
    cws_obj.generate_cws()
    cws_obj.slide_thumbnail()
    cws_obj.param()
    cws_obj.final_scan_ini()
    cws_obj.clust_tile_sh()
    cws_obj.da_tile_sh()
else:
    cws_obj = cws_czi.CWSGENERATOR(output_dir=output_dir, file_name=filename, input_dir=foldername)
    cws_obj.generate_cws()
    cws_obj.slide_thumbnail()
    cws_obj.param()
    cws_obj.final_scan_ini()
    cws_obj.clust_tile_sh()
    cws_obj.da_tile_sh()
