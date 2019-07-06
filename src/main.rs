use std::fs::File;
use std::io::prelude::*;

fn main() {
    read_all();
}

fn read_all() -> (Vec<(u8, Vec<u8>)>, Vec<(u8, Vec<u8>)>, Vec<String>) {
    let data_files = vec![
        "../cifar-10/cifar-10-batches-bin/data_batch_1.bin",
        "../cifar-10/cifar-10-batches-bin/data_batch_2.bin",
        "../cifar-10/cifar-10-batches-bin/data_batch_3.bin",
        "../cifar-10/cifar-10-batches-bin/data_batch_4.bin",
        "../cifar-10/cifar-10-batches-bin/data_batch_5.bin",
    ];
    let test_file = vec![
        "../cifar-10/cifar-10-batches-bin/data_batch_5.bin",
    ];
    let label_file = "../cifar-10/batches.meta.txt";

    let labels = read_labels(label_file);
    let train = read_files(data_files);
    let test = read_files(test_file);

    (train, test, labels)
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

fn read_labels(file: &str) -> Vec<String> {
    let mut contents = String::new();
    let mut file = File::open(file).unwrap();
    file.read_to_string(&mut contents).expect("error reading file");
    contents.lines().map(ToOwned::to_owned).collect()
}

