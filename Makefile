EXE = frostburn

default:
	EVALFILE=$(EVALFILE) cargo rustc --release -p frostburn-uci -- -C target-feature=+popcnt,+lzcnt,+bmi1
	cp target/release/frostburn-uci $(EXE)

profile:
	EVALFILE=$(EVALFILE) cargo rustc --profile profile -p frostburn-uci -- -C target-feature=+popcnt,+lzcnt,+bmi1
	cp target/profile/frostburn-uci $(EXE)
