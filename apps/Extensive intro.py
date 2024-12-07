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
