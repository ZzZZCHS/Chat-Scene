ID_format = "<OBJ{:03}>"

obj_caption_prompt = [
    "Compose a paragraph detailing the characteristics of the item.",
    "Craft a summary outlining the features of the object.",
    "Write a narrative describing the attributes of the item.",
    "Develop a report detailing the specifications of the object.",
    "Formulate a description highlighting the qualities of the item.",
    "Generate a passage outlining the characteristics of the object.",
    "Construct a statement detailing the attributes of the item.",
    "Produce a paragraph summarizing the features of the object.",
    "Draft a narrative highlighting the qualities of the item.",
    "Create a report detailing the specifications of the object.",
    "Formulate a description outlining the characteristics of the item.",
    "Compose a summary outlining the features of the object.",
    "Write a narrative detailing the attributes of the item.",
    "Develop a report highlighting the qualities of the item.",
    "Generate a paragraph describing the specifications of the object.",
    "Construct a statement outlining the characteristics of the item.",
    "Produce a summary detailing the features of the object.",
    "Draft a description highlighting the qualities of the item.",
    "Create a report outlining the specifications of the object.",
    "Formulate a narrative detailing the attributes of the item."                                                
]

obj_caption_wid_prompt = [
    "Portray the visual characteristics of the <id>.",
    "Detail the outward presentation of the <id>.",
    "Provide a depiction of the <id>'s appearance.",
    "Illustrate how the <id> looks.",
    "Describe the visual aspects of the <id>.",
    "Convey the physical attributes of the <id>.",
    "Outline the external features of the <id>.",
    "Render the appearance of the <id> in words.",
    "Depict the outward form of the <id>.",
    "Elaborate on the visual representation of the <id>."
]

multi3dref_prompt = [
    "Is there any object that matches the given description: \"{}\"? If yes, please list the IDs of all the matched objects."
]

region_caption_prompt = [
    "Describe the area surrounding {}.",
    "Provide a description of the locality around {}.",
    "Characterize the zone centered on {}.",
    "Depict the surroundings of {}.",
    "Illustrate the region with {} at its core.",
    "Give a portrayal of the area focused around {}.",
    "Offer a depiction of the vicinity of {}.",
    "Summarize the setting adjacent to {}.",
    "Explain the environment encircling {}.",
    "Detail the sector that encompasses {}."
]

scanrefer_prompt = [
    "Based on the details provided, \"<description>\", kindly specify the ID of the object that best corresponds to this information.",
    "As per the provided description, \"<description>\", could you please indicate the ID of the object that most accurately fits this description?",
    "According to the description provided, \"<description>\", please specify the ID of the object that closely resembles this description.",
    "In light of the given description, \"<description>\", would you mind providing the ID of the object that aligns most closely with this description?",
    "Considering the details given, \"<description>\", please state the ID of the object that closely matches this description.",
    "With the provided description, \"<description>\", please identify the ID of the object that closely resembles this description.",
    "Given the description, \"<description>\", could you please indicate the ID of the object that best matches this description?",
    "According to the description provided, \"<description>\", please answer with the ID of the object that closely matches this description.",
    "Based on the provided details, \"<description>\", kindly specify the ID of the object that most closely aligns with this description.",
    "Considering the given description,\"<description>\", please provide the ID of the object that best corresponds to this information."
]

scan2cap_prompt = [
    "Begin by detailing the visual aspects of the <id> before delving into its spatial context among other elements within the scene.",
    "First, depict the physical characteristics of the <id>, followed by its placement and interactions within the surrounding environment.",
    "Describe the appearance of the <id>, then elaborate on its positioning relative to other objects in the scene.",
    "Paint a picture of the visual attributes of <id>, then explore how it relates spatially to other elements in the scene.",
    "Start by articulating the outward features of the <id>, then transition into its spatial alignment within the broader scene.",
    "Provide a detailed description of the appearance of <id> before analyzing its spatial connections with other elements in the scene.",
    "Capture the essence of the appearance of <id>, then analyze its spatial relationships within the scene's context.",
    "Detail the physical characteristics of the <id> and subsequently examine its spatial dynamics amidst other objects in the scene.",
    "Describe the visual traits of <id> first, then elucidate its spatial arrangements in relation to neighboring elements.",
    "Begin by outlining the appearance of <id>, then proceed to illustrate its spatial orientation within the scene alongside other objects."
]