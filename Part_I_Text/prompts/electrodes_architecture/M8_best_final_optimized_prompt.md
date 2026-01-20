After reading the scientific publication full text provided above, **extract** the gate, source, drain materials **including any dopant element specified** and the structure‑design type that are **explicitly stated** in the text, and generate a JSON file **(with fields `gate`, `source`, `drain`, and `structure_design_type`)**. **The `gate` field must contain the material of the gate electrode (or reference electrode for electrolyte‑gated devices), not any functionalizing molecules or catalysts.** *gate*: the material of the **reference electrode** (or gate electrode for non‑electrolyte devices) as described in the paper. **For extended‑gate (remote) configurations, the gate material is the base metal of the extended gate electrode itself (e.g., the copper pad on a PCB), not the sensing membrane or functional layer attached to it.** If the paper only mentions a gate‑related material such as a catalyst or substrate, use that material for the gate field. **Normalize material names to canonical forms** (e.g., “Si substrate with HfO2” → “phosphorus/silicon”; “Ti/Au” → “titanium/gold”). If any required field (gate, source, drain, structure_design_type) is not explicitly stated, fill them with defaults: gate = "phosphorus/silicon", source = "gold", drain = "gold", structure_design_type = "Standard". **Note:** For electrolyte‑gated devices, the gate material is the reference electrode (e.g., Ag/AgCl; if given by a common name such as calomel, express it as the chemical pair ‘mercury/mercury chloride’); for BioFETs the gate material is the functional membrane composition (e.g., polypyrrole/urease). The source and drain are the channel contact metals (e.g., Ni/Au). **Clarification:** The “gate” field refers to the material of the reference electrode (or gate electrode for solid‑state gates). For back‑gated devices, the gate electrode is the substrate; extract its material from the substrate description (e.g., “n‑doped Si” → “boron/silicon”). **Include any dopant name preceding the substrate (e.g., “phosphorus/silicon”); if none is mentioned, use just the substrate material.** **If the text explicitly states that the substrate serves as the gate (e.g., “heavily phosphorus‑doped silicon wafer (the substrate) serves as the bottom‑gate electrode”), use the substrate material as the gate.** **For source and drain, extract the semiconductor channel material of the FET as explicitly described (e.g., silicon, poly‑Si, metal‑oxide, organic semiconductor), not the contact metal unless the semiconductor itself is not specified.** If the paper mentions more than one distinct FET configuration (different gate materials, source/drain metals, or structural designs), create a separate record for each configuration.  

**Structure Design Type** must be selected from the valid values list, for example:  
- *Remote (e.g., the gate electrode is physically separated from the transistor channel, as in an Extended Gate FET – EGFET)*  
- *Electrolyte‑Gated (e.g., a gate electrode in direct contact with an electrolyte)*  
- *Top‑Gated*  
- *Back‑Gated*  
- *Planar*  
- *Vertical*  
- *Coplanar*  
- *Interdigitated*  

**Decision rule:** If the gate electrode is a distinct, physically separate electrode immersed in the same electrolyte as the channel (i.e., the gate does **not** sit directly on the channel surface), label the `structure_design_type` as **'Remote'**. If the electrolyte itself serves as the gate dielectric directly contacting the channel surface (e.g., ion‑gel or liquid electrolyte on top of the channel), label it as **'Electrolyte‑Gated'**. Use the remaining terms as defined in the original list.  

For instance:  
{
  "records": [
    {
      "gate": "silver/silver chloride",
      "source": "silver",
      "drain": "silver",
      "structure_design_type": "Floating"
    }
    // continue if the publication has recorded multiple different device architectures
  ]
}