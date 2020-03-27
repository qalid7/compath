-------------------------------------------------------------
Stain Normalisation Toolbox for Matlab
Nicholas Trahearn and Adnan Khan
BIALab, Department of Computer Science, University of Warwick
-------------------------------------------------------------


-------------------------------------------------------------
0. Contents
-------------------------------------------------------------

    1. Version History
    2. System Requirements
    3. Usage Instructions
    4. Disclaimer
    5. Contact Information


-------------------------------------------------------------
1. Version History
-------------------------------------------------------------

    2.0 --- Released 26th February 2015
        - Replaced executable for Non-Linear (Khan) Stain Normalisation with a MATLAB implementation. 
        - Added functions to train custom classifiers for the Non-Linear (Khan) method (please refer to SCDTraining/README.txt for more details).
        - Removed redundant functions (AddThirdStainVector.m, Lab2RGB.m, and RGB2Lab.m) and replaced any references to them in code with the equivalent MATLAB built-in functions.
        - Separated Deconvolve.m into two functions: 
                Deconvolve.m            - Serves the same purpose as previously.
                PseudoColourStains.m    - Converts grayscale stain channels into pseudo-colour images, with respect to a given matrix. 
                                          All visualisation from Deconvolve.m is now made by a call to PseudoColourStains.m.
        - Renamed all Stain Normalisation functions and files to follow a consistent format (Norm______.m, where _____ is the name of the normalisation method).
        - Renamed the Macenko stain matrix estimation function and file to follow a consistent format for stain matrix estimation (EstUsing_______.m, where ______ is the stain matrix estimation method).

    1.0 --- Released 4th September 2014
        - Original release, containing MATLAB functions for RGB Histogram Specification, Reinhard, and Macenko methods of Stain Normalisation. 
        - Also includes an executable for the Non-Linear (Khan) Stain Normalisation method.


-------------------------------------------------------------
2. System Requirements
-------------------------------------------------------------

    In order to ensure that all features of the Toolbox function as intended we recommend the following:
        - Matlab version 2014a or later.
        - Image Processing Toolbox and Statistics Toolbox installed.
        - C compiler configured for use in Matlab.


-------------------------------------------------------------
3. Usage Instructions
-------------------------------------------------------------

    (1) Run demo.m to perform a demonstration of the Normalisation methods within the Toolbox.

    (2) The toolbox contains a MATLAB implementation of the Non-Linear Stain Normalisation algorithm reported in the following publication:

            Khan, A.M., Rajpoot, N., Treanor, D., Magee, D., A Non-Linear Mapping Approach to Stain Normalisation in Digital Histopathology Images using Image-Specific Colour Deconvolution, IEEE Transactions on Biomedical Engineering, 2014. 

        This method can be run from the MATLAB function NormSCD, please view SCDTraining/README.txt for more information about the supervised component of this method.

    (3) The toolbox also provides MATLAB implementations of three more stain normalisation algorithms generally used in histological image analysis:

        a. RGB Histogram Specification (MATLAB function NormRGBHist).
        b. Reinhard Colour Normalisation (MATLAB function NormReinhard).
        c. Macenko Stain Normalisation (MATLAB function NormMacenko).

    (4) The Toolbox also contains Matlab compatable C code for G. Landini's implementation of Ruifrok & Johnston's Colour Deconvolution algorithm. 
        Precompiled mex files for Windows and Linux are provided. This code can be also compiled manually with the command:
            mex colour_deconvolution.c
        Please ensure that your C compiler is properly configured when attempting to compile this code. This portion of the demo may not function correctly if your compiler is not correctly configured.


-------------------------------------------------------------
4. Disclaimer
-------------------------------------------------------------

    The software is provided 'as is' with no implied fitness for purpose. 
    The author is exempted from any liability relating to the use of this software.  
    The software is provided for research use only. 
    The software is explicitly not licenced for re-distribution (except via the websites of Leeds and Warwick Universities).


-------------------------------------------------------------
5. Contact Information
-------------------------------------------------------------

    Please send all comments and feedback to Nicholas Trahearn at: N.Trahearn@warwick.ac.uk

