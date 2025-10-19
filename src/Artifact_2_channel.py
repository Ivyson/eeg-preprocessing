with open("../datasets/Artifact_2_channel.TXT", "r", encoding="utf8") as file:
    data = file.read()
lines = data.strip().split("\n")

hexidecimal_chars = []
hexi = []
time_stampts = []
for line in lines:
    # print(line)
    # print()
    time_stampts.append(line.split("]")[0][1:])
    hexi.append(line.split("IN")[1].strip().split(" "))
decimals = [[int(x, 16) for x in sublist] for sublist in hexi]
with open("../datasets/Artifact_2_channel_Decimals.TXT", "w", encoding="utf8") as file:
    for i in range(len(decimals)):
        file.write(f"{time_stampts[i]} -> {' '.join(map(str, decimals[i]))}\n")
    print("Done Writing on the file")
