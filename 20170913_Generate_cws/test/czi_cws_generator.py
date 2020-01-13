import javabridge as jv, bioformats as bf
from xml.etree import ElementTree as ETree
import os
import numpy as np
import math
from PIL import Image
import pickle


class CWSGENERATOR:
    def __init__(self,
                 input_dir=os.getcwd(),
                 file_name='Test_file.svs',
                 output_dir=os.path.join(os.getcwd(), 'cws'),
                 cws_objective_value=20,
                 cws_read_size_w=2000,
                 cws_read_size_h=2000):
        jv.start_vm(class_path=bf.JARS)
        self.input_dir = input_dir
        self.file_name = os.path.basename(file_name)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self.output_dir = os.path.join(output_dir, self.file_name)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        self.bf_read = bf.ImageReader(path=os.path.join(self.input_dir, self.file_name), perform_init=True)
        self.cws_objective_value = np.int(cws_objective_value)
        self.cws_read_size = np.array([cws_read_size_w, cws_read_size_h])
        md = bf.get_omexml_metadata(path=os.path.join(self.input_dir, self.file_name))
        self.mdroot = ETree.fromstring(md)
        self.objective_power = int(float(self.mdroot[1][2].attrib['NominalMagnification']))

    def generate_cws(self):

        bf_read = self.bf_read
        cws_objective_value = self.cws_objective_value
        cws_read_size = self.cws_read_size

        if self.objective_power == 0:
            self.objective_power = int(float(self.mdroot[1][2].attrib['NominalMagnification']))

        rescale = np.int(self.objective_power / cws_objective_value)
        bf_read_size = np.multiply(cws_read_size, rescale)
        slide_h = bf_read.rdr.getSizeY()
        slide_w = bf_read.rdr.getSizeX()
        cws_h = bf_read_size[0]
        cws_w = bf_read_size[1]

        iter_tot = 0
        output_dir = self.output_dir
        text_output = open(os.path.join(output_dir, 'Output.txt'), 'w')
        for h in range(int(math.ceil((slide_h - cws_h) / cws_h + 1))):
            for w in range(int(math.ceil((slide_w - cws_w) / cws_w + 1))):
                start_h = h * cws_h
                end_h = (h * cws_h) + cws_h
                start_w = w * cws_w
                end_w = (w * cws_w) + cws_w
                if end_h > slide_h:
                    end_h = slide_h

                if end_w > slide_w:
                    end_w = slide_w
                #
                im = bf_read.read(rescale=False, XYWH=(start_w, start_h, end_w - start_w, end_h - start_h))
                format_str = 'Da%d:  start_w:%d, end_w:%d, start_h:%d, end_h:%d, width:%d, height:%d'
                text_output.write((format_str + '\n') % (
                    iter_tot, start_w, end_w, start_h, end_h, end_w - start_w, end_h - start_h)
                                  )
                print(format_str % (
                    iter_tot, start_w, end_w, start_h, end_h, end_w - start_w, end_h - start_h), flush=True)
                temp = np.array(im)
                temp = temp[:, :, 0:3]
                im = Image.fromarray(temp)
                if rescale != 1:
                    im = im.resize(size=[np.int((end_w - start_w) / rescale), np.int((end_h - start_h) / rescale)],
                                   resample=Image.BICUBIC)
                im.save(os.path.join(output_dir, 'Da' + str(iter_tot) + '.jpg'), format='JPEG')
                iter_tot += 1

        text_output.close()

    def slide_thumbnail(self):
        bf_read = self.bf_read
        cws_objective_value = self.cws_objective_value
        output_dir = self.output_dir

        if self.objective_power == 0:
            self.objective_power = int(float(self.mdroot[1][2].attrib['NominalMagnification']))
        slide_dimension = [bf_read.rdr.getSizeX(), bf_read.rdr.getSizeY()]
        rescale = np.int(self.objective_power / cws_objective_value)
        slide_dimension_20x = np.array(slide_dimension) / rescale
        scale_h = int(slide_dimension[0]) / 1024
        thumbnail__height = int(slide_dimension[1] / scale_h)
        thumb = bf_read.read(rescale=False, series=4)
        thumb = Image.fromarray(thumb)
        thumb.save(os.path.join(output_dir, 'Ss1.jpg'), format='JPEG')
        slide_thumb = thumb.resize([1024, thumbnail__height])
        slide_thumb.save(os.path.join(output_dir, 'SlideThumb.jpg'), format='JPEG')

    def param(self):
        exp_dir = self.output_dir
        if self.objective_power == 0:
            self.objective_power = int(float(self.mdroot[1][2].attrib['NominalMagnification']))
        objective_power = self.objective_power
        slide_dimension = [self.bf_read.rdr.getSizeX(), self.bf_read.rdr.getSizeY()]
        cws_objective_value = self.cws_objective_value
        rescale = np.int(objective_power / cws_objective_value)
        filename = self.file_name
        cws_read_size = self.cws_read_size

        param = {'exp_dir': exp_dir,
                 'objective_power': objective_power,
                 'slide_dimension': slide_dimension,
                 'rescale': rescale,
                 'cws_objective_value': cws_objective_value,
                 'filename': filename,
                 'cws_read_size': cws_read_size}
        pickle.dump(param, open(os.path.join(exp_dir, 'param.p'), 'wb'))

    def final_scan_ini(self):
        cws_objective_value = self.cws_objective_value
        if self.objective_power == 0:
            self.objective_power = int(float(self.mdroot[1][2].attrib['NominalMagnification']))
        rescale = np.int(self.objective_power / cws_objective_value)
        output_dir = self.output_dir
        cws_read_size = self.cws_read_size
        slide_dimension = [self.bf_read.rdr.getSizeX(), self.bf_read.rdr.getSizeY()]
        slide_dimension = np.array(slide_dimension, dtype=np.int) / rescale
        slide_dimension = slide_dimension.astype(np.int)
        slide_h = slide_dimension[1]
        slide_w = slide_dimension[0]
        cws_h = cws_read_size[0]
        cws_w = cws_read_size[1]
        text_output = open(os.path.join(output_dir, 'FinalScan.ini'), 'w')
        text_output.write('[Header]\n'
                          'iVersion=1.1DigitalSLideStudioLinux\n'
                          'tOperatorID=d1a91842-dec4-4a0a-acd2-1e4a47cae935\n'
                          'tTimeOfScan=%DATE%\n'
                          'tWebSlideTitle=\n'
                          'lXStageRef=50896\n'
                          'lYStageRef=32666\n'
                          )
        text_output.write('iImageWidth=%d\n' % (self.cws_read_size[0]))
        text_output.write('iImageHeight=%d\n' % (self.cws_read_size[1]))
        text_output.write('lXStepSize=8000\n'
                          'lYStepSize=8000\n'
                          'lXOffset=0\n'
                          'lYOffset=0\n'
                          'dMagnification=%MAGNIFICATION%\n'
                          'tImageType=.jpg\n'
                          'iFinalImageQuality=80\n'
                          'iAnalysisImageCount=%AMOUNTOFTILES%\n'
                          'iCalibrationImageCount=0\n'
                          'AIL=12.0.14 \n'
                          'bHasThumb=1\n'
                          'bHasMacro=1\n'
                          'bHasLabel=1\n'
                          'iLayers=1\n'
                          'iLevels=%AMOUNTOFLEVELS%\n'
                          'tMPP=0.496900\n'
                          )
        text_output.write('tDescription=Aperio Image Library v12.0.14   %dx%d (%dx%d) JPEG '
                          'Q=80;Aperio Image Library vFS90 01  58880x48433 [0,100 57448x48333] (256x256) JPEG/RGB '
                          'Q=70|AppMag = 20|StripeWidth = 1840|ScanScope ID = SS5306|Filename = 90284|'
                          'Date = 05/03/13|Time = 16:03:31|Time Zone = GMT+01:00|'
                          'User = d1a91842-dec4-4a0a-acd2-1e4a47cae935|Parmset = COVERSLIP|MPP = 0.4969|'
                          'Left = 25.313839|Top = 24.087423|LineCameraSkew = 0.000794|LineAreaXOffset = 0.008057|'
                          'LineAreaYOffset = -0.005012|Focus Offset = -0.000500|DSR ID = panacea|ImageID = 90284|'
                          'Exposure Time = 109|Exposure Scale = 0.000001|DisplayColor = 0|OriginalWidth = 58880|'
                          'OriginalHeight = 48433|ICC Profile = ScanScope v1\n'
                          % (slide_dimension[0], slide_dimension[1], self.cws_read_size[0], self.cws_read_size[1]))
        text_output.write('tTotalFileSize=624223872\n'
                          '[Level0]\n'
                          'iZoom=1\n'
                          )
        text_output.write('iWidth=%d\n'
                          'iHeight=%d\n'
                          % (slide_dimension[0], slide_dimension[1]))
        text_output.write('iQuality=80\n'
                          '[Level1]\n'
                          'iZoom=4\n'
                          'iWidth=14362\n'
                          'iHeight=12083\n'
                          'iQuality=90\n'
                          '[Level2]\n'
                          'iZoom=16\n'
                          'iWidth=3590\n'
                          'iHeight=3020\n'
                          'iQuality=95\n')
        iter_tot = 0
        for h in range(int(math.ceil((slide_h - cws_h) / cws_h + 1))):
            for w in range(int(math.ceil((slide_w - cws_w) / cws_w + 1))):
                x_text = ((slide_dimension[0] / 2) - (w * cws_w + cws_read_size[0] / 2)) * 4
                y_text = ((slide_dimension[1] / 2) - (h * cws_h + cws_read_size[1] / 2)) * 4
                text_output.write('[Da%d]\n'
                                  'x=%d\n'
                                  'y=%d\n'
                                  'z=0\n' % (iter_tot, x_text, y_text))
                iter_tot += 1

        text_output.close()

    def clust_tile_sh(self):
        output_dir = self.output_dir
        filename = self.file_name
        text_output = open(os.path.join(output_dir, 'clustTile.sh'), 'w')
        text_output.write('\n'
                          '#BSUB -J "%s"\n'
                          '#BSUB -o output/%s.%%J\n'
                          '#BSUB -e errors/%s.%%J\n'
                          '#BSUB -n 1\n'
                          '#BSUB -P DMPYXYAAO\n'
                          '#BSUB -W 15:00\n'
                          'startRowByRowTile.sh\n'
                          'startFinalTile.sh\n'
                          % (filename, filename, filename))
        text_output.close()

    def da_tile_sh(self):
        output_dir = self.output_dir
        cws_objective_value = self.cws_objective_value
        cws_read_size = self.cws_read_size
        if self.objective_power == 0:
            self.objective_power = int(float(self.mdroot[1][2].attrib['NominalMagnification']))
        rescale = np.int(self.objective_power / cws_objective_value)
        openslide_read_size = np.multiply(cws_read_size, rescale)
        slide_dimension = [self.bf_read.rdr.getSizeX(), self.bf_read.rdr.getSizeY()]
        slide_h = slide_dimension[1]
        slide_w = slide_dimension[0]
        cws_h = openslide_read_size[0]
        cws_w = openslide_read_size[1]
        text_output = open(os.path.join(output_dir, '_daTile.sh'), 'w')
        text_output.write('montage -limit area 12192 -limit memory 8192 \\ \n')
        iter_tot = 0
        for h in range(int(math.ceil((slide_h - cws_h) / cws_h + 1))):
            for w in range(int(math.ceil((slide_w - cws_w) / cws_w + 1))):
                text_output.write('Da%d.jpg \\ \n' % iter_tot)
                iter_tot += 1

        text_output.write('-mode Concatenate -tile  %dx%d Da_tiled.jpg \n'
                          % (int(math.ceil((slide_w - cws_w) / cws_w + 1)),
                             int(math.ceil((slide_h - cws_h) / cws_h + 1)))
                          )
        text_output.close()