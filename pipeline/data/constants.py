from pipeline.data.enums import Language

DRIVE_IDS = {
    "alaroye_mato_10k.tsv": "1FEms5rBkskG29-VkoCfFnmO6M0BLgyp-",
    "von_mato_6k.tsv": "1-nuwfHCXlecUmyvjVMZyaFubQjnohA7k",
    "masakhanews_1k.tsv": "1vmlR77KtjeeaZtfabUe_RVevIEWnDux-",
    "igbo_mato_3k.tsv": "1T_Eg6WtSCvtpNGe0QUOw78YD80Rg56xm",
    "hausa_mato_81k.tsv": "1qJFVfcJuy58oUEfZtDHqaWbbX1Ya6pxb",

    "yoruba_train_dataset.jsonl": "1awujWSzpG4FuPDssQH8oHTmMeIBmlluf",
    "yoruba_eval_dataset.jsonl": "1ZixVbGvuzA0cvgiOipgkOif9Gpx308O8",
    "yoruba_test_dataset.jsonl": "1Q8l5N2fdiVMm56Mw23YcuKtITatUFxXS",

    "yoruba_gemma3_27b_train_results.jsonl": "1s3Zjo_Ce6bJuBdQoAp5wTRylTGbZRyqY",
    "yoruba_gemma3_27b_eval_results.jsonl": "1-2eP5BDrGSnNLQGTg9E-cUgYYiFqwdys",
    "yoruba_gemma3_27b_test_results.jsonl": "1-5BlMOmBjrE_7bbGLD6WSu1Cp6x3pnuw",

    "igbo_train_dataset.jsonl": "10RHg1qWjopgjo0Ns0TZ53zhhAmVuO6u4",
    "igbo_eval_dataset.jsonl": "1tYUm3UfQyc6j6KCYhlTTQIo--sjuPn3B",
    "igbo_test_dataset.jsonl": "131_1cNei3XxlS72Ja_EHVKizB03ZEIPn",

    "igbo_gemma3_27b_train_results.jsonl": "1s1HiXac0vk9-fDMptqSeTRt6uUIXDnUc",
    "igbo_gemma3_27b_eval_results.jsonl": "1-2ni83c1kD1x6L6TitWUiRMEwyg6wUfu",
    "igbo_gemma3_27b_test_results.jsonl": "1-L3Y0IjE1qWRBZ2s2jD9_GlCHO4xmEDt",

    "hausa_train_dataset.jsonl": "1-00-aP-Vyli4MOYhr3tSiVuOyohLoShB",
    "hausa_eval_dataset.jsonl": "1q0oN33CusRs51D14P1JmzGdny6jyZxuM",
    "hausa_test_dataset.jsonl": "1OM_GP-69cJvEVdfvVixEo8aN5aGaNQqk",

    "hausa_gemma3_27b_train_results.jsonl": "1j2JlRs8LHyU9AH8IEmE2uJ4uBOkWswkb",
    "hausa_gemma3_27b_eval_results.jsonl": "1-91Uhd_R2UlH2f0qiwxz_9ssEw3fP_J9",
    "hausa_gemma3_27b_test_results.jsonl": "1PVNNLQtZdHT9j89CqqTTsmkzSbNMhzfJ",

    "filtered_yoruba_train_dataset.jsonl": "1M1YTH2jYJ6zL8T_k4Icxe7RkwlBcBB9d",
    "filtered_yoruba_eval_dataset.jsonl": "1-2uZ5lnxgQRS7c4BT3ozHxNIYDqm0rz5",
    "filtered_yoruba_test_dataset.jsonl": "1--w3T7vraOUZ3vuD_PF32uLzgq4qh9Ds",

    "filtered_igbo_train_dataset.jsonl": "1-0_kwYN2tjuoYeVMgLG9CufrjV3jecNy",
    "filtered_igbo_eval_dataset.jsonl": "1--QrFWe12OzR3SIBR9lMF2GtVdMjiBgS",
    "filtered_igbo_test_dataset.jsonl": "1J8FcO2F5h9a64Rb7N3QcVf9wqzaMs23p",

    "filtered_hausa_train_dataset.jsonl": "1L4QE3qnQifZH57XOmswOsiM9MVLpgxIr",
    "filtered_hausa_eval_dataset.jsonl": "1hPCaOzOzDKkOWF5XsvpmtGs-0-1_Sx2u",
    "filtered_hausa_test_dataset.jsonl": "1-8eWLDk9lWAx-OSE5tjQES89prhcn2yJ",

    "yoruba_english_gemma3_27b_train_results.jsonl": "1oZJBjbIGeYAJezZSz1m9lvymHt0as--p",
    "igbo_english_gemma3_27b_train_results.jsonl": "1FfleHR_1LT_8SHePGOdLw2sMqFz_TA2p",
    "hausa_english_gemma3_27b_train_results.jsonl": "1-36v03tVXYMZykvW20qmLeEPH5Vce-Hf",

    "yoruba_english_gemma3_27b_eval_results.jsonl": "1-2KtZ9CTQmmUcrtmzPeUTEtKnw-XWiZY",
    "igbo_english_gemma3_27b_eval_results.jsonl": "1--hiYF6D89_GyxvNQ_WfT_w4l5Jjvbk8",
    "hausa_english_gemma3_27b_eval_results.jsonl": "1pMUp_gBJvThw2jBoe5mlt4VO6Prv_wm8",

    "yoruba_english_gemma3_27b_test_results.jsonl": "1-R1M5MlPGLwlUOBIKxAYT6XI9SZsR2lj",
    "igbo_english_gemma3_27b_test_results.jsonl": "1--jAmqtEbki_hCQOpk_uXH-27aP00_pf",
    "hausa_english_gemma3_27b_test_results.jsonl": "13hBAJN4FazrCmpN2GGRl1kO3aDG6vLQQ",

    "filtered_english_train_dataset.jsonl": "1-9s4lsREcIemnva5yMzsTng2cwyvUk0n",
    "filtered_english_eval_dataset.jsonl": "1nFsb1y9RpvHDfrpw1JS_9Mrq3vFUzF6_",
    "filtered_english_test_dataset.jsonl": "1UDAxYEGOXRLMjEp9me3iAiuKKvUXwlwv",
}

WURA_LANG_ID_MAP = {
    Language.hausa: "hau",
    Language.igbo: "ibo",
    Language.yoruba: "yor",
}
