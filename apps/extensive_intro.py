import streamlit as st

def app():
    st.title('Basic intro')

    st.write(''' 
         Wastewater epidemiology is a new industry that aims to track vectors of disease through non-invasive means.
         There are multiple published methods of detecting disease through wastewater, but the avalible data uses a PEG centrifugal extraction method with florescent PCR detection.
         The workflow of wastewater epidemiology is as follows.
         The prosess starts by collecting a sample of wastewater from a sewer.
         A sample can be collected is one of two ways, by a composite extractor or a direct grab collection.
         A composite extractor is a collection tank that collects a small portion every few min, over the course of a day, or a grab that directly collects water from of a manhole.
         Once the sample is collected it goes to a lab for processing.
         The lab takes a 100ml portion of the 500ml sample collected and concentrates the sample by separating the water from the vector of disease.
         Contention involves trapping viruses in a net made of polyethylene glycol (PEG), removing as much water from the PEG as possible, then extracting the collected viruses from the PEG pellet.
         Once the viruses are extracted, the sample is nutralized to deactivate any virons that might be active.
         a small portion of the extraction is used for RNA concentration using a silica salt bridge extraction method.
         Once the RNA is extracted, PCR with TaqMAN probes tailord to the target gene of intrest is use and data is recorded using, qPCR or ddPCR.
         The qPCR method collects a CT value that is compared to a standerd curve of a known gene count per CT.
         The other detection method id ddPCR which produces a more direct gene count for data.
         ddPCR isolates genetic material in a PCR sample into oil wells, that idealy contains one target gene per oil well.
         The oil sepration occures before the sample enters the thermocycler and dectedtion happons after the cycles are complete.
         The completed sample plate enters a detector that runs the volumn of sample through a small tube and counts the number of oil wells present in a sample, as well as the wells floresents.
         Both CT and GC data are back calculated to estimate the amount of viral gene copys in the origanl 100ml sample fraction.
         The problem this dashboard is trying to explore is how to validate the viral gene count data to sample fecal contamination and public instancis of spicifc disease.
          ''')
    st.title('What is PMMoV?')
    st.write('''
        A big part wastewater epidemiology is controling how the environment changes the detection of viruses in wastewater.
        Studying the environments effect on the disease of interest in nearly imposable due to the true instance of a particuler disease in a population is never truly known.
        Instead of studying the effect of the environment on the disease directly, we can indirectly infer how the environment affects disease detection through a fecal matter variable.
        Currently there are a number of proposed and established fecal matter markers, (HBFs), in literature from detection of human genes, bacterial genes, surface antigens, viruses, and caffein concentration.
        The fecal matter control in our data is a plant virus called pepper mild monotilo virus or PMMoV for short.
        PMMoV spreads from peppers, or processed pepper spices to human through consumption.
        PMMoV endures the human digestive tract and is harmlessly expelled through our fecal matter where it enters the water and eventually infects more pepper plants.
        Because PMMoV is expelled through human waste, PMMoV concentration has been found to be strongly positively correlated to human waste, and because pepper consumption is common in America most human waste contains PMMoV.
        PMMoV also has the added benefit of being a virus, so PMMoV should exibits similer dynamic propertys as the other virusis we want to track.
        Although plant viruses have unique morphology compared to human viruses PMMoV is suspected of behaving similarly to viruses of interest during the collection, extraction and detection processes.
        For all of the reasons listed above, PMMoV detection is interpreted as the same as human fecal contamination data.
        The higher the PMMoV counts, the more fecal matter in a sample, the higher the suspected count of disease.
        If PMMoV counts change with environmental factors, then the suspected count of disease will positively corelate with the change in PMMoV.
        The goal of this data is to show if PMMoV contamination is  positive correlation to human diseases that can be track through waste water.
''')

    st.image('TMV.png', caption="This is an image of the Tobaco Mosaic virus, a close ansester of PMMoV, PMMoV and TMV are both rod shaped virused. Image was taken from https://www.semanticscholar.org/paper/The-physics-of-tobacco-mosaic-virus-and-virus-based-Alonso-G%C3%B3rzny/3177b81019a98aa9c2a17be46f325d1033f96f13")
