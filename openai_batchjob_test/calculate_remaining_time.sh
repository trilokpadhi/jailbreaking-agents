# 1) grab the “completed” count at t0
c0=$(curl -s https://api.openai.com/v1/batches/batch_68008730508481909bc5e84265dd2512 \
     -H "Authorization: Bearer $OPENAI_API_KEY_TP" \
     | jq .request_counts.completed)    # completed so far :contentReference[oaicite:0]{index=0}
t0=$(date +%s)

# 2) wait a bit (e.g. 30 sec)…
sleep 30

# 3) grab it again at t1
c1=$(curl -s https://api.openai.com/v1/batches/batch_68008730508481909bc5e84265dd2512 \
     -H "Authorization: Bearer $OPENAI_API_KEY_TP" \
     | jq .request_counts.completed)
t1=$(date +%s)

# 4) compute rate = Δcompleted / Δtime
delta=$((c1 - c0))
dt=$((t1 - t0))
rate=$(echo "scale=4; $delta / $dt" | bc)   # requests per second

# 5) compute how many are left
total=11250
remaining=$(( total - c1 ))

# 6) estimate seconds remaining = remaining / rate
eta=$(echo "scale=1; $remaining / $rate" | bc)

echo "Rate: $rate req/sec — Remaining: $remaining requests"
echo "≈ $eta seconds to finish"
