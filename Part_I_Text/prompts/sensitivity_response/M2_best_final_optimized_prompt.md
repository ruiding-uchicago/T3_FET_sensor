After reading the scientific publication full text provided above, please try to generate a formatted JSON file for extraction of important information. Leave "" for not available fields. Extract the reported `response_time` values (including units) and the `recovery_time` value (the numeric value that follows the phrase ‘recovery time (’, including units; also accept phrasing such as “recovery is about X‑sec” or “recovery … X sec”). **If a response or recovery time is described as “less than X s” (or similar phrasing), record it as “X s” (numeric) in the JSON.** Also locate **any reported quantitative sensitivity metric** (e.g., percent change per concentration unit, CNP shift per ppm, etc.). **If the sensitivity is given as a percent change per concentration unit (e.g., % per ppm),** – for humidity sensors use “% RH⁻¹” (percent change per percent relative humidity), and for gas sensors use “% per ppm” or “% per ppb” (percent change per ppm or ppb of the target gas). Fill **sensitivity_numerator** with the percentage (including the % sign) and **sensitivity_denominator** with the concentration value and unit. **Only extract the values for the condition with the highest NO₂ detection sensitivity (2‑PDT0.75).**  

**If the paper reports multiple sensors, polymerizing cycles, or experimental conditions, select the sensor/condition that shows the best overall performance (i.e., the highest NO₂ detection sensitivity, 2‑PDT0.75) and output **only one** record for that chosen sensor.**  

When the paper presents a sensitivity table (e.g., Supplementary Table 1), **locate the table titled ‘Detection limits and corresponding sensitivities (ΔG/G0) of various sensing materials towards NO₂’ and extract the row labeled ‘Our work’; use the ‘ΔG/G0 (%)’ value as `sensitivity_numerator` and the ‘Lower detection limit (ppb)’ (converted to ppm) as `sensitivity_denominator`.**  

For instance:
{
  "records": [
    {
      "response_time": "xx s",
      "recovery_time": "xx s",
      "sensitivity_numerator": "(the combined sensitivity value and unit as a single string, e.g., `1.19% per ppm`)",
      "sensitivity_denominator": "(leave empty if the sensitivity is provided in the numerator)"
    }
    (continue if the publication has recorded multiple different performance measurements)
  ]
}
**Definitions:** `sensitivity_numerator` – the reported percentage change (e.g., 6.68 %). `sensitivity_denominator` – the corresponding concentration unit (e.g., 1 ppm).

**When extracting sensitivity, locate the sentence that expresses a percentage change per concentration unit (e.g., “1.19% per ppb”). Record the numeric percentage **including the % sign** as `sensitivity_numerator` and the concentration unit (e.g., `ppb`) as `sensitivity_denominator`.**  

**If a sensitivity is expressed as `<value> <unit>/per<denom>` or `<value> <unit>/<denom>` (e.g., “45.9 mV/pUrea”), set `sensitivity_numerator` to the part before the slash (e.g., “45.9 mV”) and `sensitivity_denominator` to “1 pUrea”.**  

**Additional rule:** If the paper describes a stepwise change in drain current (I_D) when the pH is switched between buffers, compute the sensitivity as the absolute current change per pH unit and set `sensitivity_denominator` to `1 pH`. Likewise, if sensitivity is reported in a “value unit/pH” format (e.g., “44.86 mV/pH”), place the value with its unit (e.g., “44.86 mV”) into `sensitivity_numerator` and set `sensitivity_denominator` to `1 pH`. When a sensitivity is written as `<value> <unit>/<denominator>`, split it into two fields: `sensitivity_numerator` receives the `<value> <unit>` part (e.g., “3.43 mA”) and `sensitivity_denominator` receives the denominator expressed as “1 <short‑form of denom>” (e.g., “1 dec” for “/decade”, “1 pH” for “/pH”). **Specifically, drop any trailing “/” and write the denominator as “1 <short‑form>” (e.g., “decade” → “dec”, “pH” stays “pH”).**  

**If the paper does not explicitly mention a precise response or recovery time, but gives an approximate description (e.g., ‘less than 10 s’, ‘≈ 10 s’, ‘about 10 s’), interpret it as the nearest integer number of seconds (e.g., ‘10 s’). If no time information is provided at all, return an empty string for the corresponding field.**  

**Return the JSON object without additional markdown wrappers (i.e., a single ```json … ``` block if you wish, but no nested fences).**