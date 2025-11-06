#!/usr/bin/env bash

PORT=""
BAUD=9600
OUTFILE="e23035_mhv4_currents.csv"
INTERVAL = 1 # seconds between readings

# Configure the serial port ('8 bit, no parity, 1 stoppbit')
stty -F "$PORT" "$BAUD" cs8 -cstopb -parenb -icanon -echo

# Test to see if the serial connection is configured properly
stty -F "$PORT" -a
cat < "$PORT"

# Create csv header
if [ ! -f "$OUTFILE" ]; then
  echo "timestamp,RI0,RI1,RI2,RI3" > "$OUTFILE"
fi

echo "Logging current readings from $PORT to $OUTFILE..."
echo "Press Ctrl+C to stop."

# Read current every second
while true; do
  # Send each RI command and read the response
  RI0=$(echo -e "ri 0\r" > "$PORT"; sleep 0.1; head -n 1 < "$PORT")
  RI1=$(echo -e "ri 1\r" > "$PORT"; sleep 0.1; head -n 1 < "$PORT")
  RI2=$(echo -e "ri 2\r" > "$PORT"; sleep 0.1; head -n 1 < "$PORT")
  RI3=$(echo -e "ri 3\r" > "$PORT"; sleep 0.1; head -n 1 < "$PORT")

  TS=$(date +"%Y-%m-%d %H:%M:%S")

  echo "$TS,$RI0,$RI1,$RI2,$RI3" >> "$OUTFILE"

  sleep "$INTERVAL"
done