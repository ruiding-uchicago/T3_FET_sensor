After reading each scientific publication full text provided above, please try to generate formatted JSON file for extraction of important information. **Provide the JSON object directly, without any surrounding code fences or markdown formatting.** From the supplied article, locate and record any quantitative thickness values given for the substrate—defined as the base wafer material (e.g., Si/SiO₂) on which the device is built (e.g., µm). **The substrate may be mentioned by its full name or a common abbreviation (e.g., PET → polyethylene terephthalate); map any abbreviation to the full material name in the output.** **Do not use channel or top‑gate materials as the substrate.**—and for the dielectric layer (e.g., nm). Include these numbers (with units) in the corresponding fields. **If the paper provides a thickness for the substrate or dielectric layer, extract the exact numeric value together with its unit (e.g., “500 µm”, “100 nm”) and place it in the appropriate field; use the standard fallback values only when no thickness is mentioned.** **If a specific piece of information cannot be found in the supplied text, do not infer or guess a value; however, for `substrate_thickness` and `dielectric_layer_thickness` insert the common values for a degenerately‑doped Si wafer (≈ 525 µm) and the SiO₂ gate oxide (≈ 300 nm) instead of leaving the field empty.** For all other missing fields, set the value to an empty string (`""`) and do not fabricate any information. When several thickness numbers appear, assign each to the correct field. Expand chemical abbreviations (e.g., g‑C₃N₄ → graphitic carbon nitride, ZrO₂ → zirconium oxide) before filling the JSON fields.

**Identify the exact chemical name of the channel material as it appears in the text (including any prefixes or abbreviations, e.g., “2‑pyridyl diketopyrrolopyrrole (2‑PDT)”).**

For instance:
{
  "records": [
    {
      "substrate": "(substrate material name)",
      "substrate_thickness": "xx µm",
      "channel": "(channel/active layer material name)",
      "dielectric_layer": "(dielectric layer material name)",
      "dielectric_layer_thickness": "xx nm",
      "surface_functionalization": "(surface modification material or molecule name)",
      "structure_dimensionality": "(0D/1D/2D/3D, describing nanostructure dimensionality)"
    }
    (continue if the publication has recorded multiple different material configurations)
  ]
}
**If the device is described as a planar thin‑film transistor, film, or layer, set `structure_dimensionality` to `"2D"`. If the text mentions a conventional field‑effect transistor (e.g., ISFET, MOSFET, ISFET pH sensor) on a planar substrate, also set `structure_dimensionality` to `"2D"`. If the description indicates a bulk or three‑dimensional structure, set it to `"3D"`. If dimensionality cannot be inferred, leave the field empty (`""`).**