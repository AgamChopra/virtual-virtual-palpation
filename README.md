<h1>Virtual-Virtual Palpation</h1>
<p>Computer Vision AI framework for characterizing brain stiffness, utilizing structural data to accurately identify abnormal brain structures, including tumors. This innovative approach aims to expedite and enhance the clinical diagnosis process significantly. The proposed methodology is designed to seamlessly integrate into existing MRI protocols, ensuring a fast, efficient, and effective diagnostic pipeline.</p>
<p><h2>What is MRE?</h2>
    Magnetic Resonance Elastography (MRE) is an advanced medical imaging technique that maps the stiffness of tissues in the body. This process is particularly useful in identifying abnormalities in soft tissues, which can be an indicator of diseases. The key steps involved in MRE, are as follows:

<ol>
<li>Generation of Harmonic Shear Waves: The first step in MRE involves generating harmonic shear waves within the region of interest (ROI) in the body. This is done using an external actuator. The actuator vibrates at a steady frequency, sending shear waves through the tissue.

<li>Acquisition of MR Images: Once the shear waves are generated, Magnetic Resonance Imaging (MRI) is used to capture images of these waves as they move through the tissue. This is done by using motion-encoding gradients in the MRI, which are synchronized with the shear wave generation. The motion-encoding gradients allow the MRI to capture the tiny displacements caused by the shear waves.

<li>Image Processing to Create Elastograms: The final step involves processing the acquired MRI images to interpret the data. This process translates the observed shear wave movements into quantifiable measurements of tissue stiffness. The result is a detailed map of tissue stiffness, known as an elastogram. This step involves Non-Linear Inversion, which is an extremely time-consuming (hours to days) and computationally expensive algorithm, creating a significant bottleneck in the identification of tissue stiffness.
</ol>
Elastograms provide valuable information about the mechanical properties of tissues, which can be crucial for diagnosing various conditions, such as liver fibrosis, brain tumors, and other disorders where tissue stiffness is altered. By integrating MRE into standard MRI protocols, clinicians can gain a more comprehensive understanding of tissue health and disease without the need for invasive procedures.
</p>

<h2>Preliminary Results</h2>
<p align="center">
    <img height="550" src="https://github.com/AgamChopra/virtual-virtual-palpation/blob/main/assets/unet_run1_1000eps_1e-4_12-04-2023/Screenshot from 2024-02-09 15-10-39.png">
    <br><i>Fig. Stiffness map output of 3D-UNet model.</i><br><br>
</p>

<p><a href="https://raw.githubusercontent.com/AgamChopra/virtual-virtual-palpation/main/LICENSE" target="blank">[GNU AGPL3 License]</a></p>
