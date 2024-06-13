EXE = frostburn

default:
	RUSTFLAGS='-C target-cpu=native' EVALFILE=$(EVALFILE) cargo build --release -p frostburn-uci
	cp target/release/frostburn-uci $(EXE)
