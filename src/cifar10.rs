use std::fs::File;
use std::io::prelude::*;

pub fn read_train_data() -> (Vec<u8>, Vec<Vec<u8>>) {
    read_files_unzipped(train_files())
}

pub fn read_test_data() -> (Vec<u8>, Vec<Vec<u8>>) {
    read_files_unzipped(test_file())
} 

pub fn read_labels_data() -> Vec<String> {
    read_labels(label_file())
}

pub fn read_all() -> (Vec<(u8, Vec<u8>)>, Vec<(u8, Vec<u8>)>, Vec<String>) {
    (read_files(train_files()), read_files(test_file()), read_labels(label_file()))
}

fn read_files(files: Vec<&str>) -> Vec<(u8, Vec<u8>)> {
    files.iter().map(|f| read_file(f)).flatten().collect()
}

fn read_file(file: &str) -> Vec<(u8, Vec<u8>)> {
    let mut buffer = Vec::new();
    let mut file = File::open(file).expect("no file");
    file.read_to_end(&mut buffer).expect("error reading");
    buffer.chunks(3072)
        .map(|c| (c[0], c[1..].to_vec()))
        .collect()
}

fn read_files_unzipped(files: Vec<&str>) -> (Vec<u8>, Vec<Vec<u8>>) {
    files.iter().map(|f| read_file(f)).flatten().unzip()
}

fn read_labels(file: &str) -> Vec<String> {
    let mut contents = String::new();
    let mut file = File::open(file).unwrap();
    file.read_to_string(&mut contents).expect("error reading file");
    contents.lines().map(ToOwned::to_owned).collect()
}

fn train_files() -> Vec<&'static str> {
    vec![
        "/Users/bymarkone/Source/rust-nn/cifar10/cifar-10-batches-bin/data_batch_1.bin",
        "/Users/bymarkone/Source/rust-nn/cifar10/cifar-10-batches-bin/data_batch_2.bin",
        "/Users/bymarkone/Source/rust-nn/cifar10/cifar-10-batches-bin/data_batch_3.bin",
        "/Users/bymarkone/Source/rust-nn/cifar10/cifar-10-batches-bin/data_batch_4.bin",
        "/Users/bymarkone/Source/rust-nn/cifar10/cifar-10-batches-bin/data_batch_5.bin",
    ]
}

fn test_file() -> Vec<&'static str> {
    vec![
        "/Users/bymarkone/Source/rust-nn/cifar10/cifar-10-batches-bin/test_batch.bin",
    ]
}

fn label_file() -> &'static str {
    "/Users/bymarkone/Source/rust-nn/cifar10/cifar-10-batches-bin/batches.meta.txt"
}
