from collections import defaultdict

def HitRate(testData, predictions, n=5):
    hits = 0
    total = 0
   
    topN = GetTopN(testData, predictions, n)

    positives = testData[testData[:, 2] > 0]
    for user in topN:
        hit = False
        for topics, predictions in topN[user]:
            if topics in positives[positives[:, 0] == user, 1]:
                hit = True
                break
        if hit:
            hits += 1
        total += 1
    print(f"hits/total = {hits}/{total}")
    return hits/total


def GetTopN(testData, predictions, n=5):
    topN = defaultdict(list)
    for idx in range(testData.shape[0]):
      
        if predictions[idx] >= 0:
            user = testData[idx, 0]
            topic = testData[idx, 1]
       
            topN[int(user)].append((int(topic), predictions[idx]))

    for user, topics in topN.items():
        topics.sort(key=lambda x: x[1], reverse=True)
        topN[int(user)] = topics[:n]

    return topN