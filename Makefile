EXE = frostburn

default:
	RUSTFLAGS='-C target-cpu=native' cargo build --release -p frostburn-uci
	cp target/release/frostburn-uci $(EXE)
