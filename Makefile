EXE = frostburn

pgo:
	rm -rf target/pgo-data
	EVALFILE=$(EVALFILE) cargo rustc --release -p frostburn-uci -- -C target-feature=+popcnt,+lzcnt,+bmi1 -C profile-generate=target/pgo-data
	target/release/frostburn-uci bench
	target/release/frostburn-uci bench
	target/release/frostburn-uci bench
	target/release/frostburn-uci bench
	target/release/frostburn-uci bench
	llvm-profdata merge -o target/pgo.profdata target/pgo-data
	EVALFILE=$(EVALFILE) cargo rustc --release -p frostburn-uci -- -C target-feature=+popcnt,+lzcnt,+bmi1 -C profile-use=target/pgo.profdata
	cp target/release/frostburn-uci $(EXE)

default:
	EVALFILE=$(EVALFILE) cargo rustc --release -p frostburn-uci -- -C target-feature=+popcnt,+lzcnt,+bmi1
	cp target/release/frostburn-uci $(EXE)

profile:
	EVALFILE=$(EVALFILE) cargo rustc --profile profile -p frostburn-uci -- -C target-feature=+popcnt,+lzcnt,+bmi1
	cp target/profile/frostburn-uci $(EXE)
