use std::fs::File;
use std::io::prelude::{BufRead, Read, Seek, Write};
use std::io::{BufReader, BufWriter, Result, SeekFrom};

use cozy_chess::{Color, Move, Piece, Square};

pub struct Game {
    pub white_scharnagl: u16,
    pub black_scharnagl: u16,
    pub fake_moves: u8,
    pub color_flipped: bool,
    pub winner: Option<Color>,
    pub moves: Vec<Move>,
}

pub struct Header {
    pub wins: u64,
    pub losses: u64,
    pub draws: u64,
    pub nominal_positions: u64,
}

pub struct DataWriter {
    file: BufWriter<File>,
    header: Header,
    finished: bool,
}

impl DataWriter {
    pub fn new(file: File) -> Result<Self> {
        let mut file = BufWriter::new(file);
        let header = Header {
            wins: 0,
            losses: 0,
            draws: 0,
            nominal_positions: 0,
        };
        header.write(&mut file)?;
        Ok(DataWriter {
            file,
            header,
            finished: false,
        })
    }

    pub fn write_game(&mut self, game: &Game) -> Result<()> {
        match game.winner {
            Some(Color::White) => self.header.wins += 1,
            Some(Color::Black) => self.header.losses += 1,
            None => self.header.draws += 1,
        }
        self.header.nominal_positions += game.moves.len() as u64 - game.fake_moves as u64;
        game.write(&mut self.file)
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn finish(mut self) -> Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        self.header.write(&mut self.file)?;
        self.file.flush()?;
        self.finished = true;
        Ok(())
    }
}

impl Drop for DataWriter {
    fn drop(&mut self) {
        assert!(self.finished, "finish your data writer pls!");
    }
}

pub struct DataReader {
    file: BufReader<File>,
    header: Header,
}

impl DataReader {
    pub fn new(file: File) -> Result<Self> {
        let mut file = BufReader::new(file);
        Ok(DataReader {
            header: Header::read(&mut file)?,
            file,
        })
    }

    pub fn read_game(&mut self) -> Result<Option<Game>> {
        if self.file.fill_buf()?.is_empty() {
            return Ok(None);
        }
        Game::read(&mut self.file).map(Some)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        self.header = Header::read(&mut self.file)?;
        Ok(())
    }

    pub fn header(&self) -> &Header {
        &self.header
    }
}

impl Header {
    fn write(&self, to: &mut impl Write) -> Result<()> {
        to.write_all(&self.wins.to_le_bytes())?;
        to.write_all(&self.losses.to_le_bytes())?;
        to.write_all(&self.draws.to_le_bytes())?;
        to.write_all(&self.nominal_positions.to_le_bytes())
    }

    fn read(from: &mut impl Read) -> Result<Self> {
        let mut read = || -> Result<_> {
            let mut buf = [0; 8];
            from.read_exact(&mut buf)?;
            Ok(u64::from_le_bytes(buf))
        };
        Ok(Header {
            wins: read()?,
            losses: read()?,
            draws: read()?,
            nominal_positions: read()?,
        })
    }

    pub fn count(&self) -> u64 {
        self.draws + self.wins + self.losses
    }

    pub fn elo(&self) -> Option<(f64, f64)> {
        if self.wins == 0 || self.draws == 0 || self.losses == 0 {
            return None;
        }

        let n = self.count() as f64;
        let w = self.wins as f64;
        let d = self.draws as f64;
        let l = self.losses as f64;

        let mu = (w + 0.5 * d) / n;
        let var = (w * (1.0 - mu).powi(2) + d * (0.5 - mu).powi(2) + l * mu.powi(2)) / n;

        let mu_max = mu + 1.959963984540054 * var.sqrt() / n.sqrt();

        let elo = -400.0 * (1.0 / mu - 1.0).log10();
        let elo_max = -400.0 * (1.0 / mu_max - 1.0).log10();

        Some((elo, elo_max - elo))
    }
}

impl Game {
    fn write(&self, to: &mut impl Write) -> Result<()> {
        to.write_all(&self.white_scharnagl.to_le_bytes())?;
        to.write_all(&self.black_scharnagl.to_le_bytes())?;
        to.write_all(&[self.fake_moves | (self.color_flipped as u8) << 7])?;
        match self.winner {
            Some(Color::White) => to.write_all(&[2])?,
            Some(Color::Black) => to.write_all(&[0])?,
            None => to.write_all(&[1])?,
        }
        assert!(!self.moves.is_empty());
        for (i, mv) in self.moves.iter().enumerate() {
            let packed = mv.from as u16
                | (mv.to as u16) << 6
                | mv.promotion.map_or(6, |p| p as u16) << 12
                | ((i == self.moves.len() - 1) as u16) << 15;
            to.write_all(&packed.to_le_bytes())?;
        }

        Ok(())
    }

    fn read(to: &mut impl Read) -> Result<Self> {
        let mut buf = [0; 2];
        to.read_exact(&mut buf)?;
        let white_scharnagl = u16::from_le_bytes(buf);

        to.read_exact(&mut buf)?;
        let black_scharnagl = u16::from_le_bytes(buf);

        to.read_exact(&mut buf)?;
        let fake_moves = buf[0] & 0x7F;
        let color_flipped = buf[0] & 0x80 != 0;
        let winner = match buf[1] {
            0 => Some(Color::Black),
            1 => None,
            2 => Some(Color::White),
            _ => unreachable!(),
        };

        let mut moves = vec![];
        loop {
            to.read_exact(&mut buf)?;
            let mv = u16::from_le_bytes(buf);
            moves.push(Move {
                from: Square::index((mv & 0x3F) as usize),
                to: Square::index((mv >> 6 & 0x3F) as usize),
                promotion: Piece::try_index((mv >> 12 & 0x7) as usize),
            });
            if mv & 0x8000 != 0 {
                break;
            }
        }

        Ok(Game {
            white_scharnagl,
            black_scharnagl,
            fake_moves,
            color_flipped,
            winner,
            moves,
        })
    }
}
