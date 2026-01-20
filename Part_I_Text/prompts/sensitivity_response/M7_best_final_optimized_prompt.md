After reading the scientific publication full text provided above, please try to generate formatted JSON file for extraction of important information. Leave "" for not available fields. **If a required field is not explicitly reported in the source text, output an empty string ("") for that field and do not invent any value.** **If a particular value (response_time, recovery_time, sensitivity_numerator, sensitivity_denominator) is not explicitly stated in the text, return an empty string ("") for that field.** **If a sensitivity value is reported in a different unit, include it only if it is expressed per pH unit; if it is given as a percentage together with a concentration (e.g., “70% under 100 ppm ethanol” or “0.7%/ppm”), set `sensitivity_numerator` to the percentage and `sensitivity_denominator` to the concentration (including units), preserving the original concentration unit **exactly as it appears in the paper** (e.g., per 1 ppb, per 1 ppm).** **When a sensitivity is expressed as a percentage *per* a concentration (e.g., “6.68 % per 1 ppm”), store the percentage part in `sensitivity_numerator` (`6.68%`) and the concentration part in `sensitivity_denominator` (`1 ppm`).** **If a sensitivity is reported per decade (e.g., “3.43 mA/decade”), treat “decade” as the denominator unit and set `sensitivity_denominator` to “1 dec” (using exactly this wording).** Ignore sensitivities related to other analytes (e.g., DNA, antibodies).** **For sensitivity, use the change in drain current (ΔI, in µA) observed when the target analyte is introduced as the numerator, and the corresponding concentration (in ppm or ppb) as the denominator.** Return the result as raw JSON text only, without any markdown code fences. Output the JSON object directly, without any markdown code fences or extra formatting.

**Specifically, extract the response time, recovery time, and sensitivity values for **the sensor with the highest reported performance** (as presented in the paper’s main performance summary, e.g., the values listed in the “Results” or “Conclusion” section where the sensor’s overall performance is highlighted). **If the paper compares multiple sensor configurations, choose the one the authors explicitly label as best (e.g., the Ga‑doped ZnO nanowire sensor (3GZO)).** **When multiple sensitivities are reported, use the sensitivity value of the Ga‑doped ZnO nanowire sensor (3GZO) as the target.** **When multiple values are reported, select the central (non‑±) figures that correspond to the sensor’s fastest response and recovery highlighted in the summary (e.g., 4 s response, 36 s recovery) and the percentage sensitivity given for the primary analyte concentration (e.g., 69 % at 20 ppm).** 

Provide a JSON object containing a record for the selected sensor with the fields response_time, recovery_time, sensitivity_numerator, and sensitivity_denominator. **Your output should be a JSON object … with fields response_time, recovery_time, sensitivity_numerator, and sensitivity_denominator.** Locate the sentence in the paper that reports the sensor’s sensitivity (e.g., “Target sensitivity = 52.8 mV/pH”) and split it: put the numeric value with its unit into `sensitivity_numerator` (e.g., “52.8 mV”) and set `sensitivity_denominator` to “1 pH”. **Your response must be a JSON object with a key `records` that holds an array containing exactly one record object. Each record should have the following keys: `response_time`, `recovery_time`, `sensitivity_numerator`, `sensitivity_denominator`.** Include only the available record.

**Definition of sensitivity fields:**  
• **sensitivity_numerator** – the reported change in sensor signal (as a percentage % or a factor) when exposed to the target analyte.  
• **sensitivity_denominator** – the corresponding concentration (in ppm, ppb, or %).  

**Formatting rule:** `sensitivity_numerator` must contain the numeric value **followed by its unit** (e.g., "58.3 µA" or "7.9 %"). `sensitivity_denominator` must be expressed as a unit without extra wording (e.g., "1 ppm", "1 ppb", "1 pH").

*Example:* a reported “7.9 % increase at 1 ppm NO₂” should be recorded as `sensitivity_numerator`: "7.9 %", `sensitivity_denominator`: "1 ppm".

For instance:
{
  "records": [
    {
      "response_time": "xx s",
      "recovery_time": "xx s",
      "sensitivity_numerator": "(the change in drain current (ΔI) when pH increases by one unit, e.g., µA or mA)",
      "sensitivity_denominator": "(always \"1 pH\")"
    }
  ]
}