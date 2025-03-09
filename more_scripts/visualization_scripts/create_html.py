# Define classes and corresponding video filenames
rows = [
    ("Pyramid Flow (SD3 backbone)", ["violin_sd3_orig.mp4", "dog_sd3_orig.mp4", "gun_sd3_orig.mp4"]),
    ("Another Class", ["example1.mp4", "example2.mp4", "example3.mp4"])  # Add more as needed
]

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
            <th>LTX-Video + LoRa_V ✅ + LoRa_A ❌</th>
            <th>LTX-Video + LoRa_V ✅ + LoRa_A ✅</th>
        </tr>
"""

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
                    <source src="pyramid/{video_file}" type="video/mp4">
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
file_path = "./samples/temp.html"
with open(file_path, "w", encoding="utf-8") as file:
    file.write(html_content)

print(f"HTML file saved as {file_path}")
