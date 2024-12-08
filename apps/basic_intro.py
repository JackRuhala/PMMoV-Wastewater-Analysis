import streamlit as st

def app():
    st.title('Basic intro')

    st.write(''' 
        Humans are a complex ecosystem of viruses, bacteria, and cells all sharing one body. 
        Sometimes viruses and bacteria don’t play nicely with our bodies and make us sick.
        While our bodies have some cool defense systems against disease, our infections can still slip past our defense and often can be found in unexpected places.
        One such place our illnesses can wine up is in our poop.
        poop, being covered in germs is not a new idea, but testing poop for specific infections is a resent development.
        However if an individual want to know if they have a disease or not, there are normally easier ways to test that then sending in a poop sample to a lab.
        While poop testing might be inefficient for an individual, what about testing a population?
        Americans are lucky enough to have a sewer system that carries away the waste we don’t like to think about, but there’s valuable information about disease in that sewer water.
        Wastewater epidemiology targets the collective fecal matter in the sewer system and tests the water for specific diseases.
        Wastewater epidemiology provides a way to gain valuable knowledge about a disease in an environment without needing to test all individuals in a population.
          ''')
    st.title('The Problem With Wastewater Epidemiology')
    st.write ('''
        One question people want to know about disease tracking is, how many people have a particuler disease?
        Wastewater epidemiology, unfortently has a hard time guadging the scale of an outbrake, Why?
        Imagin a cup of water and add a few drops of food coloring to the water, then give the water to a friend and ask them how much food coloring is in the water.
        Your friend could tell you what color the water is or how vibrint the color of the water is, but geting an exact amount of food coloring based on the color alone is very hard.
        Your freind might respond with vuage words like, not much, alot, or moderate in regards to how much food coloring was added but they most likly will never get the exact amount of color added right.
        Wastewater epidemiology works the same way as the food coloring problem, exept the food coloring is a particuler disease, and the water is always changing.
        To get a closer estiment of how much disease is in the water, you will want to know somthing about the water first.
        Things to look for in the water are its tempature, its pH, how fast is the water moveing, how much sediment is in the water, ect.
        One importent question epidemiology has is how much poop is in the water.
        By the time the water is colected all fecal mater is completly disintigrated so you can tell if your sample has what you want or not by eye.
        To guage fecal contamination of the water, you collect the water and messure a property of the water that come from poop.
        There are many ways to do this, but the data we have uses a virus called PMMoV.
        PMMoV is a hardy plant virus that can be found in peppers or anything made from peppers.
        How PMMoV is dectected is given in more detail in the extensive intro, but its not importent here.
        Whats importent to know is the more PMMoV is detected, the more poop is in the water, the more likely we find a disease in the water.
        The corilation between PMMoV to poop to disease has not been relably shown in sciance yet and is the main focus of this analysis.
        Can we use PMMoV to tell us somthing about the water we are testing, and if we can, dose using PMMoV help us get a more accurate prediction of a disease in a population.
