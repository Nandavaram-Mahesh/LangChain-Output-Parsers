import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    
)

model = ChatHuggingFace(llm=llm)


# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

result1 = model.invoke(prompt2)

print(result1.content)

# <think>
# Hmm, the user has shared a report on black holes and requested a 5-line summary. First, I need to analyze this dense material about cosmic phenomena where gravity dominates completely. The original text covers formation mechanisms, structural components like singularities and event horizons, detection methods, and unresolved paradoxes.

# The key challenge is distilling this complex astrophysics into just five accessible lines while preserving scientific accuracy. I'll prioritize the core concepts: what black holes fundamentally are, how we observe them, why they matter to physics, and the remaining mysteries.

# From the report's structure, I note these essential elements to include:
# - Their defining nature as regions of no escape
# - Formation through stellar collapse leading to supermassive variety
# - Detection via accretion disks and gravitational wave signatures
# - Theoretical implications like Hawking radiation
# - Ongoing questions about information paradox and singularities

# The summary must maintain neutrality while being engaging - perhaps using vivid language like "cosmic vacuums" but avoiding sensationalism. I'll sequence chronologically from definition to research frontiers, ensuring each line delivers one key insight without technical jargon. The final point should emphasize their role in advancing physics, as the report highlights them as "laboratories for fundamental physics."

# This approach balances accessibility with the report's academic depth, serving both students needing quick review and curious readers seeking bite-sized cosmic wonders.
# </think>
# Here is a concise 5-line summary of the text on black holes:

# 1.  **Gravity's Extreme Triumph:** Black holes are cosmic voids where gravity is so intense that nothing, not even light, can escape once beyond the event horizon.  
# 2.  **Formation Pathways:** They form primarily through the collapse of massive stars (stellar-mass), collisions/massive gas cloud collapse (intermediate-mass), or accumulation over cosmic time (supermassive at galactic centers).
# 3.  **Detection & Evidence:** We detect them via orbiting stars/X-ray binaries, accretion disks, galactic core motion, gravitational waves from mergers, and direct imaging by the Event Horizon Telescope.
# 4.  **Theoretical Frontiers:** Key concepts include Hawking Radiation (evaporation via quantum effects), the unresolved Information Paradox (loss of infalling information), and the need for quantum gravity to understand singularities.
# 5.  **Cosmic Significance:** Black holes are crucial laboratories for extreme physics (testing Relativity and Quantum Mechanics), drive galaxy evolution via AGN feedback, and their study remains vital for understanding the universe's fundamental laws and history.

