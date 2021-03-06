"""
Description: classes and functions needed for the real time audio process

Author: Adrien Llave - CentraleSupélec
Date: 29/08/2018

Version: 11.0

Date    | Auth. | Vers.  |  Comments
18/03/28  ALl     1.0       Initialization
18/03/30  ALl     2.0       Bug fix in Butterworth class, remove initial click due to bad conditioning
18/04/01  ALl     3.0       Minor bug fix:  - DMA compensation gain
                                            - DMA hybrid Lowpass filter cutoff frequency high to optimize WNG
18/05/22  ALl     4.0       Add processing: - FilterBank
                                            - DMA adaptive
                                            - Multiband expander
                                            - Overlap-add block processing
18/06/18  ALl     5.0       DMA adaptive:   - bug fix
                                            - apply LP and HP filter before cross corr estimation
18/06/19  ALl     6.0       - AudioCallback: Add management of the end ('end_b' flag attribute, fill by zeros last buffer)
                            - Add processing: NoiseSoustraction
18/06/21  ALl     7.0       Add processing: - CompressorFFT
                                            - OverlapSave
18/07/02  ALl     8.0       Add processing: - BeamformerMVDR in order to replace BeamformerDAS which is wrong
18/07/10  ALl     8.1       Remove class:   - BeamformerDAS (old version of MVDR)
                                            - Noise reduction (multi-band expander)
18/07/17  ALl     8.2       BeamformerMVDR : bug fix, complex cast some variables
18/07/19  ALl     9.0       Add processing: DMA adaptive in FREQ domain
                                            bug fix: prevent division by 0 in coeff estimation
18/08/24  ALl    10.0       Add processing: SuperBeamformer (Optimal DI Beamformer)
18/08/29  ALl    11.0       - Dependency issue fixing between RTAudioProc and binauralbox
                                - remove binauralbox dependency to RTAudioProc in order to make RTAudioProc dependant to bb
                                - move RTBinauralizer and RTBinauralizerFFT from bb to rt
                                TODO: change 'bb' to 'rt' when using those classes
                            - Add security freq2time in RTBinauralizer

"""

from setuptools import setup

setup(name='RTAudioProc',
      version='11.0',
      description='classes and functions needed for the real time audio process',
      url='',
      author='Adrien Llave',
      author_email='adrien.llave@gmail.com',
      license='',
      packages=['RTAudioProc'],
      # install_requires=['numpy', 'time', 'copy'],
      zip_safe=False)
