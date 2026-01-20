After reading the scientific publication full text provided above, please try to generate a formatted JSON file for extraction of important information. Leave "" for not available fields. **Create a single JSON record for the paper; auxiliaries are not separate records.** Search both the main manuscript and any Supporting Information for each required value; include data found only in the supplementary sections. **If the paper reports multiple sensing microneedles (Na⁺, K⁺, Ca²⁺, pH), list them as a comma‑separated list in the appropriate field (e.g., surface_functionalization).** If any of the fields below are not explicitly mentioned in the paper, **infer a typical ISFET value (e.g., substrate = 'boron/silicon', substrate_thickness ≈ 525 µm, dielectric_layer = 'silicon nitride', dielectric_layer_thickness ≈ 1 nm) and use it; if no reasonable typical value can be assumed, leave the field empty ("")**. **If a numeric substrate thickness is mentioned, look for a numeric value followed by a unit (µm, nm, etc.) that is associated with the substrate and extract that number, reporting it in micrometers (µm).** If substrate, dielectric layer, or their thicknesses are missing, for thickness fields infer a realistic typical value (e.g., ~500 µm for bulk Si wafers, ~10 nm for common gate oxides, ~2 nm for native oxides) and include it; otherwise leave the field empty (""). **Do not insert any explanatory text or placeholder values; use only an empty string ("") for missing or not applicable fields.** (If substrate is generic e.g., “glass”, or described as a wafer, board, or other form, record the material on which the devices are fabricated and any thickness provided.)

**Note:** When populating the **channel** field, list only the semiconductor material that forms the conductive path of the FET (e.g., silicon, graphene). The sensing film or MOF should be recorded under **surface_functionalization**, not as the channel.

For instance:
{
  "records": [
    {
      "substrate": "soda‑lime glass",
      "substrate_thickness": "1000 µm",
      "channel": "(channel/active layer material name)",
      "dielectric_layer": "(dielectric layer material name)",
      "dielectric_layer_thickness": "xx nm",
      "surface_functionalization": "(surface modification material or molecule name, list multiple items separated by commas)",
      "structure_dimensionality": "(0D/1D/2D/3D, describing nanostructure dimensionality)"
    }
    // (continue if the publication has recorded multiple different material configurations)
  ]
}
// The order of the keys in each record does not matter.