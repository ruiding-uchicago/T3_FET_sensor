After reading the scientific publication full text provided above, please try to generate a formatted JSON file for extraction of important information. Leave "" for not available fields, and for `substrate_thickness` and `dielectric_layer_thickness` extract the exact numeric value and unit explicitly given for the substrate or dielectric layer respectively (ignore other layer thicknesses). **If the substrate thickness is mentioned, always include it in the `substrate_thickness` field, capturing the exact numeric value and unit (e.g., “525 µm”). If not stated, apply the following defaults: silicon wafer – 525 µm; glass – 1 mm; quartz – 500 µm; flexible polymer (e.g., PET) – 125 µm; other substrates – leave the field empty.** **Note: The “channel” field should denote the sensing transistor used in the ISFET system (e.g., p‑type Si‑FET, organic TFT), not the reference‑electrode metal stack.**  

When filling the “substrate” field, **select the material that is described as the supporting bulk or wafer on which the device is built; do not treat active sensing materials or coatings as the substrate.** If the substrate is listed with multiple components separated by “/”, retain the full slash‑separated string as the substrate name; if the substrate is described as ‘degenerately doped silicon’ (or any similar phrasing), treat the substrate as plain silicon and do not infer or output the dopant element; otherwise, if the substrate is a doped semiconductor, include the dopant element (e.g., phosphorus‑doped silicon) as part of the substrate description. **If the paper mentions multiple substrates, report the substrate that hosts the transistor channel (the material on which the FET itself is built) and ignore substrates used only for auxiliary components such as reference electrodes or packaging.** **If the paper mentions a generic substrate name such as “glass”, interpret it as a silicon‑based substrate and output the material as “silicon dioxide/sodium oxide/calcium oxide”; assign a default thickness of “1000 µm” unless a specific thickness is explicitly provided.**  

For the `dielectric_layer` field, **only include the material that functions as the gate insulator in the final device; ignore any polymer or resist layers that are used solely for processing (e.g., PMMA, photo‑resist).** **Only list a dielectric layer if the paper explicitly mentions a separate gate‑dielectric material (e.g., Al₂O₃, HfO₂, Si₃N₄) distinct from any native oxide on the substrate.** If the device contains more than one layer that could be considered a dielectric, **do not treat the native oxide layer that directly contacts the channel surface as the dielectric unless it is explicitly identified as a gate dielectric**; encapsulating polymers or protective coatings should be placed under *surface_functionalization* (or omitted if not a functionalization). **Only fill *surface_functionalization* when a distinct material is intentionally added to the gate electrode surface that is different from the electrode/contact material; if the functional element is the same as the electrode metal or no separate functionalization is applied, leave the field empty.** If the substrate description includes an oxide layer (e.g., Si/SiO₂) and its thickness is provided, treat that oxide as the dielectric layer and populate `dielectric_layer` and `dielectric_layer_thickness` accordingly.  

For instance:
{
  "records": [
    {
      "substrate": "(substrate material name)",
      "substrate_thickness": "(xx µm if mentioned)",
      "channel": "(channel/active layer material name)",
      "dielectric_layer": "(dielectric layer material name)",
      "dielectric_layer_thickness": "xx nm",
      "surface_functionalization": "(surface modification material or molecule name)",
      "structure_dimensionality": "(0D/1D/2D/3D, describing nanostructure dimensionality)"
      // Create one dictionary for each sensing microneedle on the MMNs (Na⁺, K⁺, Ca²⁺, and pH). The order of the dictionaries does not matter.
    }
    (continue if the publication has recorded multiple different material configurations)
  ]
}

/* When a material is given in the paper as an abbreviation (e.g., ZrO₂, PMMA, PMF), output its **full chemical name** in the JSON (e.g., “zirconium oxide”, “poly(methyl methacrylate)”, “poly melamine co‑formaldehyde”). Preserve the order of components as they appear (e.g., “zirconium oxide/poly(methyl methacrylate)/poly melamine co‑formaldehyde”). */