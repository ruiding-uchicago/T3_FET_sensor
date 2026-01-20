Here is the JSON file capturing the hydrothermal synthesis details for the ZnO nanosheets described in the passage. After reading the scientific publication full text provided above, extract any explicitly labeled annealing (if present) and hydrothermal heating details (temperature, duration, atmosphere) and generate a formatted JSON file. If a liquid‑phase treatment (e.g., mixed acid, aqueous solution) is given with temperature and time, record them as hydrothermal_temperature and hydrothermal_time; be sure to extract the hydrothermal reaction duration (e.g., “90 °C for 1.5 h”) and place it in hydrothermal_time. Extract the relevant records and format them as JSON, filling each field only when the information is present.

For instance:
{
  "records": [
    {
      "annealing_temperature": "xx °C",
      "annealing_time": "xx h",
      "annealing_atmosphere": "(e.g. air, N2, Ar, O2, vacuum, H2)",
      "hydrothermal_temperature": "xx °C",
      "hydrothermal_time": "xx h"
    }
    (continue if the publication has recorded multiple different thermal processing conditions)
  ]
}
If the manuscript does not mention an annealing step, **do not infer or estimate**; set annealing_temperature, annealing_time, and annealing_atmosphere to empty strings (""). Likewise, if no hydrothermal step is described, **do not infer or estimate**; leave hydrothermal_temperature and hydrothermal_time empty. If no explicit values are found, **do not infer or estimate**; leave the corresponding fields empty (""), and if no annealing or hydrothermal step is described, output a list containing a single record where all fields are empty strings. **Return only a valid JSON object (no markdown code fences).**