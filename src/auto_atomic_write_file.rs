use anyhow::Result;
use atomic_write_file;
use atomic_write_file::AtomicWriteFile;
use std::io::{BufWriter, Write};
use std::path::Path;

pub struct AutoAtomicWriteFile(Option<BufWriter<AtomicWriteFile>>);

impl AutoAtomicWriteFile {
    pub fn new(overwrite: bool, p: impl AsRef<Path>) -> Result<Self> {
        let p: &Path = p.as_ref();

        anyhow::ensure!(
            overwrite || !p.exists(),
            "File {} already exists, and --overwrite option was not used",
            p.display()
        );

        let f = AtomicWriteFile::open(p)?;
        Ok(AutoAtomicWriteFile(Some(BufWriter::new(f))))
    }

    pub fn finish_keep(&mut self) -> Result<()> {
        if let Some(mut w) = self.0.take() {
            w.flush()?;
            w.into_inner()?.commit()?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        if let Some(mut w) = self.0.take() {
            w.flush()?;
            w.into_inner()?.commit()?;
        }
        Ok(())
    }
}

impl Write for AutoAtomicWriteFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self.0.as_mut() {
            Some(x) => x.write(buf),
            None => std::io::Result::Err(std::io::Error::from_raw_os_error(9)),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self.0.as_mut() {
            Some(x) => x.flush(),
            None => std::io::Result::Err(std::io::Error::from_raw_os_error(9)),
        }
    }
}

impl Drop for AutoAtomicWriteFile {
    fn drop(&mut self) {
        self.finish_keep().ok();
    }
}
