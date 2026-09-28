# Troubleshooting

**Load library does nothing, or reports an error.**

- Check **Preferences ▸ Add-ons ▸ irsim Thermal Materials ▸ Test connection**.
- *No module named numpy*: the Python set there is not the project's. Use `python.sh`, not the
  `kit/python/bin/python3` inside it, because only `python.sh` sets up its environment.

**Export is refused.** The message says why:

- below 95 % coverage;
- the name is taken by a hand-written asset;
- an earlier export exists and *Replace earlier export* is not ticked;
- the folder holds files the add-on did not write;
- a connection or hidden part does not pass the check (the message names it).

**A part stays grey in the thermal view.** It has no material at all. Assign it one and it gets a
plain material carrying the infrared one.

**The finder lists a contact between parts that do not really touch.** Downloaded parts often
overlap a little. Reject it, or lower **Touching within**.

**The finder misses two parts you know touch.** They may be further apart than the touching
distance. Raise it, or select both and click **Connect selected**.

**A connection says one of its parts is gone.** The part was deleted or joined into another.
Delete the connection and search again.
