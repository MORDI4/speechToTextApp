package com.example.basics;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class TrainingData implements Serializable {
    private List<double[]> mfccSamples; // Przechowuje próbki MFCC
    private String dataName; // Nazwa danych treningowych

    public TrainingData(String dataName) {
        this.dataName = dataName;
        this.mfccSamples = new ArrayList<>();
    }

    public void addMFCCSample(double[] mfcc) {
        mfccSamples.add(mfcc);
    }

    public void saveDataToFile(String fileName) {
        try (DataOutputStream outputStream = new DataOutputStream(new FileOutputStream(fileName, true))) {
            for (double[] sample : mfccSamples) {
                // Rozważ dodanie separatora pomiędzy próbkami MFCC dla łatwiejszego odczytu.
                outputStream.writeInt(sample.length);
                for (double value : sample) {
                    outputStream.writeDouble(value);
                }
            }
            System.out.println("Dane zostały dopisane do pliku: " + fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static TrainingData loadDataFromFile(String dataName) {
        // Rozważ dodanie walidacji pliku przed przetwarzaniem danych.
        TrainingData trainingData = new TrainingData(dataName);
        try (DataInputStream inputStream = new DataInputStream(new FileInputStream(dataName))) {
            while (inputStream.available() > 0) {
                int sampleSize = inputStream.readInt(); // Odczytaj rozmiar próbki
                if (sampleSize <= 0) {
                    System.err.println("Błąd: Nieprawidłowa długość próbki.");
                    return null; // Zakończ wczytywanie danych treningowych w przypadku nieprawidłowej długości próbki
                }
                double[] sample = new double[sampleSize];
                for (int i = 0; i < sampleSize; i++) {
                    sample[i] = inputStream.readDouble(); // Odczytaj kolejne wartości próbki
                }
                trainingData.addMFCCSample(sample);
            }
            System.out.println("Dane treningowe zostały wczytane pomyślnie: " + dataName);
            return trainingData;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    // Getters and setters
    public List<double[]> getMfccSamples() {
        return mfccSamples;
    }

    public void setMfccSamples(List<double[]> mfccSamples) {
        this.mfccSamples = mfccSamples;
    }

    public String getDataName() {
        return dataName;
    }

    public void setDataName(String dataName) {
        this.dataName = dataName;
    }
}