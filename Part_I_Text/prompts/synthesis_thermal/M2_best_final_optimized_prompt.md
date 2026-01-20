After reading the scientific publication full text provided above, please try to generate a formatted JSON file for extraction of important information. **When locating thermal information, focus on sentences that include keywords such as “annealed”, “annealing”, “grown”, “furnace”, “hydrothermal”, “temperature”, “time”, or “atmosphere”, and extract the numeric values, units, and atmosphere exactly as presented.** **Note**: *In Korean articles, annealing temperature may be expressed as “어닐링 온도”, annealing time as “시간”, and annealing atmosphere as “대기”. Use these terms to locate the corresponding values.* **Use only the information explicitly stated in the article; do not fabricate or guess any values.** If a step is not mentioned, set the corresponding fields to an empty string (`""`), i.e., leave them blank. Otherwise, extract any thermal step (annealing, **furnace growth**, curing, baking, hydrothermal) with its temperature and duration, **but for annealing **or furnace growth** only use temperatures from statements that explicitly contain the word “annealed” or “grown” (e.g., “annealed at 650 °C for 20 min …” or “grown at 800 °C for 0.5 h …”).** **If the article contains no thermal‑step description, you must still output a record, leaving the fields empty (`""`).** If the paper mentions a thermal step such as “treated in oven at X °C for Y h” (curing, baking, oven‑cure, etc.), **do not** record this as a hydrothermal step; only record hydrothermal conditions when the manuscript explicitly refers to a *hydrothermal* step (e.g., “hydrothermal synthesis at X °C for Y h”).  

**If thermal steps are described, extract their details into the appropriate fields (annealing_* **(including furnace growth)** and hydrothermal_*). If no such steps are described, you may leave those fields empty.**  

**Return exactly one record (the array must contain a single object).**  

For instance:
{
  "records": [
    {
      "annealing_temperature": "xx °C",
      "annealing_time": "xx h",
      "annealing_atmosphere": "Only if step called annealing or furnace growth; (e.g. air, N2, Ar, O2, vacuum, H2; if presented as “Ar/H₂” or “Ar, H₂”, list each gas separately, separated by commas)",
      "hydrothermal_temperature": "xx °C",  // temperature of the explicitly mentioned hydrothermal step
      "hydrothermal_time": "xx h"           // duration of the explicitly mentioned hydrothermal step
    }
  ]
}
If any of these values are missing, you should provide empty strings for that field in the output. **If the annealing time is reported in minutes, convert it to hours (1 hour = 60 minutes) and round to four decimal places before placing it in the `annealing_time` field.**