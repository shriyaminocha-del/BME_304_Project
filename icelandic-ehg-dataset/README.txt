Short name: ice16ehgdb
Long name: Icelandic 16-electrode Electrohysterogram (EHG) Database
Creators of the Dataset: Asgeir Alexandersson, Thora Steingrimsdottir, Jeremy Terrien, Catherine Marque, Brynjar Karlsson
Reference: Alexandersson, A., Steingrimsdottir, T., Terrien, J., Marque, C., Karlsson, B..The Icelandic 16-electrode electrohysterogram database. Sci. Data 2:150017 doi:10.1038/sdata.2015.17 (2015).
Contact information: Asgeir Alexandersson, alexandersson@gmail.com - alternate contact: Brynjar Karlsson, brynjar@ru.is

Icelandic 16-electrode Electrohysterogram (EHG) Database

Introduction

Preterm birth is a major health problem and the main cause of mortality and morbidity in new-borns. Studies on the electrohysterogram (EHG) have shown promising results regarding preterm labour prediction. A sixteen electrode recording system in a 4-by-4 configuration on the abdomen can provide information on the propagation of uterine electrical activity.

This database consists of 122 16-electrode EHG recordings performed on 45 pregnant women. The recordings were performed between 2008 and 2010 in Iceland. Pregnancy recordings, defined as recordings performed at antenatal care clinics on participants in the third trimester and not suspected to be in labour, were performed at Akureyri Primary Health Care Centre and Landspitali University Hospital. Labour recordings, defined as measurements performed on participants suspected to be in labour, present in the labour wards and who delivered within 24 hours, were performed at Landspitali University Hospital and Akureyri Hospital. Participants had normal singleton pregnancies and no known risk factors for preterm birth. After each pregnancy recording, the participant was invited to take part in another recording one to two weeks later. The protocol was approved by the National Bioethics Committee in Iceland (VSN 02-006-V4).

The measurements were performed using a sixteen channel multi-purpose physiological signal recorder (Embla A10), most commonly used for investigating sleep disorders. An anti-aliasing filter with a high cut-off frequency of 100 Hz was used but no high pass filter was used. The signal sampling rate was 200 Hz and the signal was digitized to 16 bits. The sixteen monopolar electrode signals were originally stored in the EDF (European Data Format) format by the Somnologica software used to control the Embla A10.

A standardized 4-by-4 electrode EHG recording protocol, using reusable Ag/AgCl electrodes with a 13.0 mm outer diameter and an 8.0 mm inner diameter, were used for the recordings. The distance between electrode centres was 17.5 mm. The electrode numbering scheme, as seen when looking at the abdomen of the participant, can be seen in the figure below (Figure 1).

The desired position on the abdomen was with the third vertical line of electrodes (electrodes 9 to 12) placed on the median axis of the uterus and the 10th-11th pair of electrodes half way between the uterine fundus and pubic symphysis. The navel was avoided by displacing the matrix up or down whilst staying as close as possible to the desired position. A patient ground electrode and a reference electrode were placed over the iliac crests on either side (sides not standardized). The figure below (Figure 2) shows the electrode positions. The corner electrodes of the 4-by-4 grid are labelled and the black electrodes represent the patient ground and reference electrodes.

A tocodynamometer was attached to the abdomen during recordings. When possible, a support, such as a pillow, was positioned under the right side of the participant to prevent potential aortocaval compression syndrome. During a recording, the researcher (AA) recorded participant movements, equipment manipulation, participant-reported possible contractions and foetal movements, and any other unusual events. The participant and researcher conversed freely during the recordings and no restrictions were placed on the participant in terms of changing position if needed.

The intended duration of a pregnancy recording was one hour (actual average recording time was 61 minutes, range 19-86 minutes) and the intended duration of a labour recording was at least half an hour (actual average recording time was 36 minutes, range 8-64 minutes) but the participant could stop the measurement at any time. The majority of participants were measured repeatedly during the same pregnancy and took part in two to seven recordings. The lowest gestational age was 29 weeks and 5 days (29w5d - pregnancy recording) and the highest gestational age was 41 weeks and 5 days (41w5d - labour recording). A detailed description of the recording protocol can be found in the Scientific Data article.

Database Format

File names in the database are of the form ice###_*type*_*record number*, where ice### is the ID of the participant (e.g. ice001), *type* refers to the type of recording: p (pregnancy) or l (labour), and *record number* is the number of recording for that particular participant (e.g. 1of3). Each recording has three associated files:

	-A scanned tocograph with a manually inserted recording time axis (.jgp file). Each small square represents 30 seconds.
	-A binary signal (.dat) file containing the data from the 16 monopolar electrodes.
	-A header (.hea) file.

The strings at the end of lines 2-17 in the header files correspond to the signal labels. The signals are labelled with "EHGn" where "n" refers to the relevant electrode number (electrode numbers are not in ascending order). Information from each recording is at the end of the header file and includes:

	-Participant ID
	-Record number
	-Record type (labour, pregnancy)
	-Age of participant (years)
	-BMI (body mass index) of participant before pregnancy
	-BMI of participant at time of recording
	-Gravidity (number of times participant has been pregnant, including current pregnancy)
	-Parity (previous births after 22 weeks gestation)
	-Previous caesarean (Yes, No)
	-Placental position
	-Gestational age at recording (weeks/days), according to a first trimester ultrasound 
	-Gestational age at delivery (weeks/days) 
	-Mode of delivery (Vaginal, Vaginal/Induction, Elective caesarean, Emergency caesarean due to slow progress, Emergency caesarean due to other than slow progress). Vaginal delivery indicates spontaneous onset unless appended with /Induction.
	-Synthetic oxytocin use in labour (Yes, No)
	-Epidural during labour (Yes, No)
	-Comments for recording
	-Comments for delivery

For 111 of the recordings, there is also an annotation (.atr) file containing the type and timing of events, added manually using WAVE. The types of event are:

	- C - Contraction. Used when the participant feels a contraction or there is a very likely contraction on the tocograph (not always used when there is an obvious contraction on the tocograph).
	- (c) - Possible contraction. Used when there is not a very likely contraction but the participant has pressure sensation or a contraction is suspected on the tocograph.
	- pm - Participant movement.
	- pos - Participant change of position.
	- fm - Foetal movement. Used when the participant feels foetal movement.
	- em - Equipment manipulation. Used when electrodes are pressed more firmly onto the abdomen if otherwise not explained in the comments.

The database also includes:

	-The zip file icelandic16ehgmat.zip that includes MATLAB (.mat) versions of all the signal files along with header files (file names of the form ice###_*type*_*record number*m) and MATLAB (.mat) versions of the annotations (file names of the form ice###_*type*_*record number*m_ann). The MATLAB files contain the absolute raw units (divide by 131.068 to obtain the physical units in mV). Please note that the first sample of a signal file is indexed with 1 in the .mat files but with 0 in the .dat files and so there is an index number discrepancy of one between the two file formats (.mat and .dat).
	-RECORDS.txt containing a list of the recordings by record name, with one record name per line.
	-info.txt containing the clinical information from each recording (information from the header files excluding comments) along with recording durations and whether or not a recording has a corresponding annotation file.

Usage Notes

The annotations and the tocograph complement each other. Some contractions that are present in the annotations are not obvious on the tocograph and some obvious contractions on the tocograph are not in the annotations. This explains in part why some recordings are without annotations. Noted foetal movement could last for differing amounts of time and participants did not always notify if they felt foetal movement. Sometimes foetal movement can be seen on the tocograph.

There can be differences in when participants start to feel a contraction or foetal movements and so differences in when participants notify about events. Events were therefore occasionally approximated to the nearest whole minute. There are also delays internal to the tocodynamometers. Due to factors such as these, the inserted recording times on the tocographs and the annotation times may be up to ± 30 seconds from the actual recording or event times.

Even though some pregnancies ended in caesarean, the participant was on occasion already in spontaneous labour. These incidences are explained in the comments sections of the header files. If a recording was close to birth, then the timing of the recording with regard to the birth is in the comments of the header file.