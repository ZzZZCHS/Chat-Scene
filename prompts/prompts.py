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
    "Are there any objects fitting the description of \"<description>\"? If so, kindly provide the IDs for those objects.",
    "Do any objects match the description of \"<description>\"? If they do, please share the IDs of those objects.",
    "Is there anything that matches the description \"<description>\"? If yes, please share the IDs of those objects.",
    "Are there objects that correspond to the description \"<description>\"? If there are, kindly list their IDs.",
    "Does anything fit the description of \"<description>\"? If it does, could you list the IDs for those objects?",
    "Are there objects described as \"<description>\"? If there are, please provide the IDs for those objects.",
    "Have any objects been described as \"<description>\"? If so, please share the IDs of those objects.",
    "Do any objects meet the criteria of \"<description>\"? If they do, kindly provide the IDs of those objects.",
    "Are there objects with the attributes of \"<description>\"? If there are, please list their IDs.",
    "Are there any objects that correspond to the description \"<description>\"? If yes, could you share the IDs for those objects?"
]

multi3dref_location_prompt = [
    "Are there any objects that correspond to the description \"<description>\"? If yes, could you share the locations for those objects?"
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

grounding_prompt = [
    "Share the ID of the object that best fits the description \"<description>\".",
    "Kindly provide the ID of the object that closely matches the description \"<description>\".",
    "What is the ID of the object that aligns with the description \"<description>\"?",
    "Identify the ID of the object that closely resembles the description \"<description>\".",
    "What's the ID of the object that corresponds to the description \"<description>\"?",
    "Give the ID of the object that most accurately describes the description \"<description>\".",
    "Share the ID of the object that best corresponds to the description \"<description>\".",
    "Identify the ID of the object that closely aligns with the description \"<description>\".",
    "What is the ID of the object that matches the description \"<description>\"?"
    # "According to the given description, \"<description>,\" please provide the ID of the object that closely matches this description."
]

grounding_location_prompt = [
    "According to the given description, \"<description>,\" please provide the location of the object that closely matches this description."
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

scan2cap_location_prompt = [
    "Here is an object located at <loc>. Begin by detailing the visual aspects of this object before delving into its spatial context among other elements within the scene.",
]

nr3d_caption_prompt = [
    "Detail the spatial positioning of the <id> amidst surrounding elements.",
    "Illustrate the <id>'s placement relative to its environment.",
    "Explain the <id>'s location in correlation with nearby items.",
    "Elaborate on the <id>'s spatial context within the scene.",
    "Describe how the <id> is situated in relation to other elements present.",
    "Provide insight into the <id>'s positioning among its surroundings.",
    "Discuss the relative placement of the <id> compared to its surrounding context.",
    "Offer a depiction of the <id>'s spatial orientation within the scene.",
    "Interpret the <id>'s location within the broader context of the scene.",
    "Present the <id>'s spatial relationship with other entities within the scene."
]