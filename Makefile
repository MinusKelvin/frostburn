EXE = frostburn

default:
	EVALFILE=$(EVALFILE) cargo rustc --release -p frostburn-uci -- -C target-feature=+popcnt,+lzcnt,+bmi1
	cp target/release/frostburn-uci $(EXE)
