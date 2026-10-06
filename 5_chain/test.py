class nig:
    s = 'postivie'

raw = nig.s
print(f'goti it {raw}')
if "pos" in raw:
    sentiment = "positive"
elif "neg" in raw:
    sentiment = "negative"
else:
    sentiment = "unknown"

print(sentiment)