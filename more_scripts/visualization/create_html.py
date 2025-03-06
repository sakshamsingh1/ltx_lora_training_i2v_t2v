# Define classes and corresponding video filenames
import pandas as pd

# Start the HTML content
html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Table Layout</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid black;
            text-align: center;
            padding: 10px;
        }
        img, video {
            max-width: 100%;
            height: auto;
            display: block;
            margin: auto;
        }
    </style>
</head>
<body>
    <table>
        <tr>
            <th>Class</th>
            <th>LTX-Video + LoRa_V ❌ + LoRa_A ❌</th>
            <th>LTX-Video + LoRa_V ❌ + LoRa_A ✅</th>
            <th>LTX-Video + LoRa_V ✅ + LoRa_A ✅</th>
        </tr>
"""

meta_path = '/home/sxk230060/ltx_lora_training_i2v_t2v/preprocess/asva_metadata.csv'
df = pd.read_csv(meta_path)
df = df[df['split']=='test'].reset_index(drop=True)
df = df.groupby("label").first().reset_index()
df.head()

# rows = [
#     ("baby_babbling_crying", ["no_lora/av_5Uodeo_Ln84_000014_000024_6.5_9.5.mp4", "_lora_a/av_5Uodeo_Ln84_000014_000024_6.5_9.5.mp4", "lora_v_lora_a/av_5Uodeo_Ln84_000014_000024_6.5_9.5.mp4"]),
# ]

rows = [
    ("baby_babbling_crying", ["no_lora/av_5Uodeo_Ln84_000014_000024_6.5_9.5.mp4", "_lora_a/av_5Uodeo_Ln84_000014_000024_6.5_9.5.mp4", "lora_v_lora_a/av_5Uodeo_Ln84_000014_000024_6.5_9.5.mp4"]),
    ("cap_gun_shooting", ["no_lora/av_03fGTwkSBWs_000289_000299_0.0_5.0.mp4", "_lora_a/av_03fGTwkSBWs_000289_000299_0.0_5.0.mp4", "lora_v_lora_a/av_03fGTwkSBWs_000289_000299_0.0_5.0.mp4"]),
    ("chicken_crowing", ["no_lora/av_231-4GOUaDQ_000014_000024_7.0_10.0.mp4", "_lora_a/av_231-4GOUaDQ_000014_000024_7.0_10.0.mp4", "lora_v_lora_a/av_231-4GOUaDQ_000014_000024_7.0_10.0.mp4"]),
    ("dog_barking", ["no_lora/av_2jIv5qBTS88_000009_000019_1.5_5.5.mp4", "_lora_a/av_2jIv5qBTS88_000009_000019_1.5_5.5.mp4", "lora_v_lora_a/av_2jIv5qBTS88_000009_000019_1.5_5.5.mp4"]),
    ("frog_croaking", ["no_lora/av_-INNrk4mik4_000031_000041_7.0_10.0.mp4", "_lora_a/av_-INNrk4mik4_000031_000041_7.0_10.0.mp4", "lora_v_lora_a/av_-INNrk4mik4_000031_000041_7.0_10.0.mp4"]),
    ("hammering", ["no_lora/av_0_1Jo7NAhl4_000149_000159_2.5_6.0.mp4", "_lora_a/av_0_1Jo7NAhl4_000149_000159_2.5_6.0.mp4", "lora_v_lora_a/av_0_1Jo7NAhl4_000149_000159_2.5_6.0.mp4"]),
    ("lions_roaring", ["no_lora/av_3yXCtpvjz6E_000017_000027_1.0_4.5.mp4", "_lora_a/av_3yXCtpvjz6E_000017_000027_1.0_4.5.mp4", "lora_v_lora_a/av_3yXCtpvjz6E_000017_000027_1.0_4.5.mp4"]),
    ("machine_gun_shooting", ["no_lora/av_-njqm6R9Tko_000050_000060_5.5_10.0.mp4", "_lora_a/av_-njqm6R9Tko_000050_000060_5.5_10.0.mp4", "lora_v_lora_a/av_-njqm6R9Tko_000050_000060_5.5_10.0.mp4"]),
    ("playing_cello", ["no_lora/av_6zrX3NgsL7U_000290_000300_0.0_9.5.mp4", "_lora_a/av_6zrX3NgsL7U_000290_000300_0.0_9.5.mp4", "lora_v_lora_a/av_6zrX3NgsL7U_000290_000300_0.0_9.5.mp4"]),
    ("playing_trombone", ["no_lora/av_22mgtk4Iw0M_000113_000123_0.0_10.0.mp4", "_lora_a/av_22mgtk4Iw0M_000113_000123_0.0_10.0.mp4", "lora_v_lora_a/av_22mgtk4Iw0M_000113_000123_0.0_10.0.mp4"]),
    ("playing_trumpet", ["no_lora/av_0bC2T-xZkCs_000187_000197_0.0_3.0.mp4", "_lora_a/av_0bC2T-xZkCs_000187_000197_0.0_3.0.mp4", "lora_v_lora_a/av_0bC2T-xZkCs_000187_000197_0.0_3.0.mp4"]),
    ("playing_violin_fiddle", ["no_lora/av_-j9x-d4ZqtY_000030_000040_0.0_9.5.mp4", "_lora_a/av_-j9x-d4ZqtY_000030_000040_0.0_9.5.mp4", "lora_v_lora_a/av_-j9x-d4ZqtY_000030_000040_0.0_9.5.mp4"]),
    ("sharpen_knife", ["no_lora/av_-1rZFviqTTQ_000028_000038_3.5_5.8.mp4", "_lora_a/av_-1rZFviqTTQ_000028_000038_3.5_5.8.mp4", "lora_v_lora_a/av_-1rZFviqTTQ_000028_000038_3.5_5.8.mp4"]),
    ("striking_bowling", ["no_lora/av_1gadod9llZY_000046_000056_4.5_8.5.mp4", "_lora_a/av_1gadod9llZY_000046_000056_4.5_8.5.mp4", "lora_v_lora_a/av_1gadod9llZY_000046_000056_4.5_8.5.mp4"]),
    ("toilet_flushing", ["no_lora/av_1AfCMvhZJVY_000040_000050_0.5_3.5.mp4", "_lora_a/av_1AfCMvhZJVY_000040_000050_0.5_3.5.mp4", "lora_v_lora_a/av_1AfCMvhZJVY_000040_000050_0.5_3.5.mp4"]),
]


# Dynamically add table rows
for class_name, video_files in rows:
    html_content += f"""
        <tr>
            <td>{class_name}</td>
    """
    for video_file in video_files:
        html_content += f"""
            <td>
                <video controls>
                    <source src="{video_file}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </td>
        """
    html_content += "</tr>\n"  # Close the row

# Close the HTML structure
html_content += """
    </table>
</body>
</html>
"""

# Save the HTML content to a file
file_path = "/mnt/ssd0/saksham/i2av/ltx_lora_training_i2v_t2v/outputs/temp.html"
with open(file_path, "w", encoding="utf-8") as file:
    file.write(html_content)

print(f"HTML file saved as {file_path}")
