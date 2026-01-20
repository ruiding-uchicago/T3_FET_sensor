After reading the scientific publication full text provided above, please generate a formatted JSON file extracting only annealing or hydrothermal steps that are explicitly described in the paper (or synonyms such as “heat‑treatment”, “hydrothermal synthesis”). **In the Methods section, locate the sentence that explicitly states the annealing conditions (for example, “The PNS was annealed at 200 °C for 1 h in an Argon atmosphere.”) and extract the temperature, time, and atmosphere from it.** Specifically, extract the annealing temperature, annealing time, annealing atmosphere, hydrothermal temperature, and hydrothermal time. **Note: temperatures may appear with the character ‘℃’; treat this as ‘°C’ and extract the numeric value (e.g., ‘800℃’ → temperature = 800 °C).** **If the paper reports a pulsed‑laser‑deposition‑in‑furnace (PLDF) step with a temperature and duration (e.g., “grown at 800 ℃ for 30 min”), map that temperature to `annealing_temperature` and the duration (converted to hours) to `annealing_time`; use the listed carrier gas as `annealing_atmosphere`.** **Only fill the annealing_* fields if the article contains an explicit statement that the material was “annealed” (or “annealing”, “heat‑treated”, “baked”, “cured”) **as part of the substrate‑surface‑treatment** (e.g., the H₂ anneal on the SiO₂/Si substrate) at a given temperature, time, and atmosphere. Do not treat drying, solvent‑removal, or storage steps as annealing, and ignore any annealing steps performed later on devices or for measurements. Leave "" for not available fields. If any of the five parameters is not stated in the paper, output an empty string (`""`). **Do not infer or guess values; only use information explicitly mentioned.** **Only output values that are explicitly present in the article text. If you cannot locate an exact value, leave the field empty – do not estimate, convert, or infer any numbers.**  

When multiple time specifications are given for the same step (e.g., a value in parentheses and another later), use the value inside the parentheses as the definitive time (report it exactly as shown, e.g., “0.3333 h”).  

Your output should be a JSON object with the following fields: annealing_temperature, annealing_time, annealing_atmosphere, hydrothermal_temperature, hydrothermal_time (When the paper explicitly mentions a sample being “annealed”, “treated in an oven”, “baked”, “cured”, or similar, treat it as annealing. Only treat a temperature as hydrothermal_temperature if the paper explicitly refers to a hydrothermal step or synthesis; do not infer hydrothermal conditions from ambient or storage temperatures).  

The temperature in Celsius should be a numeric value without any units (e.g., 150). The time should be expressed in hours (e.g., 2 for 2 hours). **If the source text provides the time in minutes, convert it to hours (divide by 60) and round to four decimal places before placing it in the JSON output.**  

For instance, fill in the following template with the extracted values:

```json
{
  "records": [
    {
      "annealing_temperature": "<fill‑in>",
      "annealing_time": "<fill‑in>",
      "annealing_atmosphere": "<fill‑in>",
      "hydrothermal_temperature": "<fill‑in>",
      "hydrothermal_time": "<fill‑in>"
    }
    // (continue if the publication has recorded multiple different thermal processing conditions)
  ]
}
```