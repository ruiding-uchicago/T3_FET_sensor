After reading the scientific publication full text provided above, please try to generate a formatted JSON file for extraction of important information **for the transistor of interest (e.g., the floating‑gate boron/silicon transistor) described in the paper**. **If the paper does not explicitly mention a field‑effect transistor (e.g., ISFET, MOSFET‑based sensor, BioFET, EnFET, etc.), output an empty JSON with all fields left empty.** **Extract the materials used for the gate, source, and drain electrodes exactly as they appear in the article.** **Note: `gate` = electrolyte/reference electrode material (the electrode that provides the gate voltage, not the metal gate or ion‑sensing membrane). For CS‑FET devices, this refers to the functional sensing/active layer (e.g., MOF such as HKUST‑1, ZIF‑8) deposited on the channel.** **Specifically, locate and record the gate electrode material (e.g., ITO) in the `gate` field.** If the gate material is described as a substrate or back‑gate (e.g., “phosphorus‑doped silicon substrate”), treat that substrate material as the gate material. → When the gate electrode is described as a coated layer (e.g., indium tin oxide, ITO), record that coating material as the gate (e.g., “indium tin oxide”). Use “X gate” wording to get material (e.g., “Al gate” → “aluminum”). **When the gate material is described as “X‑doped Y”, output it as “X/Y” (e.g., “phosphorus/silicon”).** Leave "Gold" for not available fields. **If the gate material is not explicitly stated (e.g., only a back‑gate substrate is mentioned), record the substrate material (e.g., silicon) or, if no substrate is given, assume it is the same as the source and drain material (gold) rather than leaving the field empty.**  

**If the paper does not mention any specific FET structure type, set the `structure_design_type` field to `"standard"`.** **If the gate is a separate off‑chip microneedle array that is electrically connected to the transistor by a flexible Ag‑NW trace (i.e., the gate is physically remote from the FET channel), set `structure_design_type` to `remote gate`; otherwise, when the gate electrode is directly fabricated on the same chip as the transistor, set it to `extended‑gate`.**

**If the source and/or drain material is not explicitly mentioned in the paper, assume they are "gold".** **If the structure design type is not explicitly mentioned, assume "standard".**

For instance:
{
  "records": [
    {
      "gate": "(gate electrode material name, i.e., the material of the electrolyte reference electrode; e.g., \"silver/silver chloride\" for an Ag/AgCl reference)",
      "source": "(source electrode material name)",
      "drain": "(drain electrode material name)",
      "structure_design_type": "(the overall layout of the transistor as described in the paper, e.g., \"source‑centred\", \"gate‑centred\", \"standard\")"
    }
    (Create one record for each distinct FET configuration (i.e., each different gate material) described in the paper)
  ]
}