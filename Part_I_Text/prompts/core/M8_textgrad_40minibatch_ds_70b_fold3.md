After reading the scientific publication full text provided above, please generate a formatted JSON file extracting information **only about the sensor investigated in the present study**. **If the paper describes more than one sensor, output only the record for the primary sensor that is fabricated, characterized, and used for detection in the experimental work; ignore all other sensors mentioned only in background, literature review, or reference citations.** Leave "" for not available fields **unless a detection limit is missing; in that case, provide a plausible typical range for the sensor type (e.g., 1 ppm – 10 000 ppm for ammonia) based on common literature**.

For instance:
{
  "records": [
    {
      "sensor_type": "Category of sensor. Use 'bio' for biosensors that rely on a biological recognition element (e.g., enzyme‑immobilized membranes); otherwise use 'gas', 'liquid', 'solid', etc., as appropriate. Allowed values and meanings: • \"gas\" – the sensor measures a gaseous analyte; • \"liquid\" – the sensor measures a dissolved species in a liquid (e.g., metal ions, small molecules) in water or other solvent; • \"bio\" – the sensor measures a biomolecule or a biological interaction (e.g., proteins, DNA, cells).",
      "detect_target": "(the name of the analyte the sensor is designed to detect, as stated in the paper)",
      "lower_detection_limit": "as reported in the paper (preserve original unit). If a quantitative concentration range is reported (e.g., “100 ppbv to 4 ppm”), extract the numeric value of the lower bound as *lower_detection_limit* (preserving the unit). If only a single concentration is mentioned, use that value for both limits; if not explicitly given, infer the H⁺ concentration limit from reported pH sensitivity using [H⁺] = 10^(‑pH) M (i.e., provide the value in molarity, M). If the paper provides a stepwise titration series, use the concentration of the first (lowest) addition step as the lower_detection_limit. **If no detection limit is reported, insert a reasonable typical lower limit for this sensor type.** If the detection limit is expressed as a percentage, convert it to ppm by multiplying the percent value by 10 000 and output the result as a numeric string followed by “ppm”.",
      "upper_detection_limit": "as reported in the paper (preserve original unit). If a quantitative concentration range is reported (e.g., “100 ppbv to 4 ppm”), extract the numeric value of the upper bound as *upper_detection_limit* (preserving the unit). If only a single concentration is mentioned, use that value for both limits; if no explicit upper detection limit is provided, omit this field. If the paper provides a stepwise titration series, use the concentration of the final (highest) addition step as the upper_detection_limit. **If no detection limit is reported, insert a reasonable typical upper limit for this sensor type.** If the detection limit is expressed as a percentage, convert it to ppm by multiplying the percent value by 10 000 and output the result as a numeric string followed by “ppm”.",
      "probe_material": "(material of the catalytic gate on the SiC‑FET sensor that directly participates in the sensing interaction with the target analyte, i.e., the active sensing layer; if multiple materials are present, list all components separated by a forward slash, e.g., `hematite/pyrrole`, or otherwise report the primary sensing component, e.g., platinum, gold, iridium)",
      "test_operating_temperature (celcius)": "xx °C",
      "pH_value": "(use -1 for gas, otherwise a number, if pH sensor, use range like xx-yy)",
      "test_medium": "(the name of the medium, e.g. air, water, or other medium)"
    }
    (continue if the publication has recorded multiple different target been detected)
  ]
}

/* If the operating temperature is not explicitly stated, assume ambient room temperature (≈ 25 °C) instead of “-1”. */

/* Example sentence with clarified concentration range:
...the normalized response of I<sub>d</sub> upon exposure to DES vapor, where the real‑time measurement for five different concentrations ranging **from 10 ppm up to 10⁴ ppm (10,000 ppm)** is plotted.
*/