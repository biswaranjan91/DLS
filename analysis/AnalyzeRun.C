// ============================================================================
//  analysis/AnalyzeRun.C
//  ROOT macro for offline analysis of NeutrinoDetSim CSV output.
//  Reads all results/<medium>_<E>MeV/summary.csv files and produces
//  publication-quality ROOT TCanvas plots saved as PDF + PNG.
//
//  Usage (ROOT):
//    root -l -b -q 'analysis/AnalyzeRun.C("./results")'
// ============================================================================

#include "TCanvas.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TColor.h"
#include "TLatex.h"
#include "TLine.h"
#include "TBox.h"
#include "TArrow.h"
#include "TFile.h"
#include "TPad.h"
#include "TAxis.h"
#include "TFrame.h"

#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <algorithm>

// ── Media list ────────────────────────────────────────────────────────────────
const std::vector<std::string> kMedia = {
  "H2O","D2O","WbLS","GdWbLS","TeWbLS","D2OLAB","D2OGdLAB","D2OTeLAB"
};
const std::vector<std::string> kMediaLabels = {
  "H_{2}O","D_{2}O","WbLS","Gd-WbLS","Te-WbLS",
  "D_{2}O+LAB","D_{2}O+Gd-LAB","D_{2}O+Te-LAB"
};
const std::vector<double> kEnergies = {0.5, 1.0, 3.0, 5.0, 10.0};

// ── Per-medium colours (ROOT colour indices) ───────────────────────────────
const std::vector<int> kColors  = {kBlue+1, kAzure-2, kOrange+7, kOrange-2,
                                   kRed+1,  kGreen+2, kGreen-3,  kViolet+1};
const std::vector<int> kMarkers = {20, 21, 22, 23, 33, 29, 34, 43};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
std::map<std::string,double> LoadSummary(const std::string& dir,
                                          const std::string& medium,
                                          double energy)
{
  std::ostringstream ss;
  ss << dir << "/" << medium << "_" << std::fixed
     << std::setprecision(1) << energy << "MeV/summary.csv";
  std::map<std::string,double> m;
  std::ifstream f(ss.str());
  if (!f.is_open()) return m;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0]=='#') continue;
    auto comma = line.find(',');
    if (comma == std::string::npos) continue;
    std::string key = line.substr(0, comma);
    std::string val = line.substr(comma+1);
    try { m[key] = std::stod(val); } catch(...) {}
  }
  return m;
}

void LoadTimingHist(const std::string& dir, const std::string& medium,
                    double energy, TH1D*& h)
{
  std::ostringstream ss;
  ss << dir << "/" << medium << "_" << std::fixed
     << std::setprecision(1) << energy << "MeV/timing_hist.csv";
  h = new TH1D(Form("hTime_%s_%.1f", medium.c_str(), energy),
               Form("%s @ %.1f MeV; Time [ns]; Counts", medium.c_str(), energy),
               500, 0, 500);
  std::ifstream f(ss.str());
  if (!f.is_open()) return;
  std::string line;
  while (std::getline(f,line)) {
    if (line.empty()||line[0]=='#') continue;
    auto c = line.find(',');
    if (c==std::string::npos) continue;
    double t = std::stod(line.substr(0,c));
    double n = std::stod(line.substr(c+1));
    h->Fill(t, n);
  }
}

void LoadWavelengthHist(const std::string& dir, const std::string& medium,
                        double energy, TH1D*& h)
{
  std::ostringstream ss;
  ss << dir << "/" << medium << "_" << std::fixed
     << std::setprecision(1) << energy << "MeV/wavelength_hist.csv";
  h = new TH1D(Form("hWL_%s_%.1f", medium.c_str(), energy),
               Form("%s @ %.1f MeV; Wavelength [nm]; Counts", medium.c_str(), energy),
               100, 250, 650);
  std::ifstream f(ss.str());
  if (!f.is_open()) return;
  std::string line;
  while (std::getline(f,line)) {
    if (line.empty()||line[0]=='#') continue;
    auto c = line.find(',');
    if (c==std::string::npos) continue;
    double wl = std::stod(line.substr(0,c));
    double n  = std::stod(line.substr(c+1));
    h->Fill(wl, n);
  }
}

void ApplyStyle()
{
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetPadTopMargin(0.08);
  gStyle->SetPadBottomMargin(0.12);
  gStyle->SetPadLeftMargin(0.13);
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetLabelFont(42,"xyz");
  gStyle->SetTitleFont(42,"xyz");
  gStyle->SetTitleSize(0.045,"xyz");
  gStyle->SetLabelSize(0.038,"xyz");
  gStyle->SetLineWidth(2);
  gStyle->SetFrameLineWidth(1);
  gStyle->SetHistLineWidth(2);
  gStyle->SetGridStyle(3);
  gStyle->SetGridColor(kGray);
  gStyle->SetPadGridX(true);
  gStyle->SetPadGridY(true);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main function
// ─────────────────────────────────────────────────────────────────────────────
void AnalyzeRun(const char* resultsDir = "./results")
{
  ApplyStyle();
  std::string rDir(resultsDir);
  std::filesystem::create_directories("root_plots");

  // Pre-load all summaries
  std::map<std::string, std::map<double, std::map<std::string,double>>> allData;
  for (const auto& m : kMedia)
    for (double e : kEnergies)
      allData[m][e] = LoadSummary(rDir, m, e);

  // ── Canvas 1: Light yield + Detected photons vs energy ────────────────────
  {
    TCanvas* c1 = new TCanvas("c1","LightYield",1400,600);
    c1->Divide(2,1);

    TMultiGraph* mg1 = new TMultiGraph("mg1",
      "Total Photons Produced;Electron Energy [MeV];Photons / Event");
    TMultiGraph* mg2 = new TMultiGraph("mg2",
      "Detected Photons;Electron Energy [MeV];Detected Photons / Event");
    TLegend* leg = new TLegend(0.15,0.60,0.50,0.92);
    leg->SetBorderSize(0); leg->SetFillStyle(0);

    for (size_t i=0; i<kMedia.size(); ++i) {
      auto g1 = new TGraph(); auto g2 = new TGraph();
      int pt=0;
      for (double e : kEnergies) {
        auto& s = allData[kMedia[i]][e];
        if (s.empty()) continue;
        g1->SetPoint(pt, e, s["mean_produced_per_event"]);
        g2->SetPoint(pt, e, s["mean_detected_per_event"]);
        ++pt;
      }
      g1->SetLineColor(kColors[i]); g1->SetMarkerColor(kColors[i]);
      g1->SetMarkerStyle(kMarkers[i]); g1->SetLineWidth(2);
      g2->SetLineColor(kColors[i]); g2->SetMarkerColor(kColors[i]);
      g2->SetMarkerStyle(kMarkers[i]); g2->SetLineWidth(2);
      mg1->Add(g1,"LP"); mg2->Add(g2,"LP");
      leg->AddEntry(g1, kMediaLabels[i].c_str(), "lp");
    }

    c1->cd(1); mg1->Draw("A"); gPad->SetLogx(); gPad->SetLogy();
    mg1->GetXaxis()->SetTitle("Electron Energy [MeV]");
    mg1->GetYaxis()->SetTitle("Photons / Event");
    TLatex tl; tl.SetNDC(); tl.SetTextFont(42); tl.SetTextSize(0.042);
    tl.DrawLatex(0.15,0.93,"Total Optical Photons Produced");
    leg->Draw();

    c1->cd(2); mg2->Draw("A"); gPad->SetLogx(); gPad->SetLogy();
    mg2->GetXaxis()->SetTitle("Electron Energy [MeV]");
    mg2->GetYaxis()->SetTitle("Detected Photons / Event");
    tl.DrawLatex(0.15,0.93,"Detected Photons at PMTs");
    leg->Draw();

    c1->SaveAs("root_plots/C1_LightYield.pdf");
    c1->SaveAs("root_plots/C1_LightYield.png");
  }

  // ── Canvas 2: Detection efficiency + Cherenkov fraction ───────────────────
  {
    TCanvas* c2 = new TCanvas("c2","Efficiency",1400,600);
    c2->Divide(2,1);

    TMultiGraph* mgEff = new TMultiGraph();
    TMultiGraph* mgCer = new TMultiGraph();
    TLegend* leg = new TLegend(0.55,0.60,0.92,0.92);
    leg->SetBorderSize(0); leg->SetFillStyle(0);

    for (size_t i=0; i<kMedia.size(); ++i) {
      auto gEff = new TGraph(); auto gCer = new TGraph();
      int pt=0;
      for (double e : kEnergies) {
        auto& s = allData[kMedia[i]][e];
        if (s.empty()) continue;
        gEff->SetPoint(pt, e, s["detection_efficiency"]*100.);
        gCer->SetPoint(pt, e, s["cerenkov_fraction_produced"]*100.);
        ++pt;
      }
      gEff->SetLineColor(kColors[i]); gEff->SetMarkerColor(kColors[i]);
      gEff->SetMarkerStyle(kMarkers[i]); gEff->SetLineWidth(2);
      gCer->SetLineColor(kColors[i]); gCer->SetMarkerColor(kColors[i]);
      gCer->SetMarkerStyle(kMarkers[i]); gCer->SetLineWidth(2);
      mgEff->Add(gEff,"LP"); mgCer->Add(gCer,"LP");
      leg->AddEntry(gEff, kMediaLabels[i].c_str(), "lp");
    }

    TLatex tl; tl.SetNDC(); tl.SetTextFont(42); tl.SetTextSize(0.042);
    c2->cd(1); mgEff->Draw("A"); gPad->SetLogx();
    mgEff->GetXaxis()->SetTitle("Electron Energy [MeV]");
    mgEff->GetYaxis()->SetTitle("Detection Efficiency [%]");
    tl.DrawLatex(0.15,0.93,"Detection Efficiency");
    leg->Draw();

    c2->cd(2); mgCer->Draw("A"); gPad->SetLogx();
    mgCer->GetXaxis()->SetTitle("Electron Energy [MeV]");
    mgCer->GetYaxis()->SetTitle("Cherenkov Fraction [%]");
    tl.DrawLatex(0.15,0.93,"Cherenkov Fraction of Produced Photons");
    leg->Draw();

    c2->SaveAs("root_plots/C2_Efficiency.pdf");
    c2->SaveAs("root_plots/C2_Efficiency.png");
  }

  // ── Canvas 3: Timing histograms (8 media at 5 MeV) ────────────────────────
  {
    TCanvas* c3 = new TCanvas("c3","Timing",1600,900);
    c3->Divide(4,2);
    for (size_t i=0; i<kMedia.size(); ++i) {
      c3->cd(i+1);
      TH1D* h = nullptr;
      LoadTimingHist(rDir, kMedia[i], 5.0, h);
      if (!h || h->GetEntries()==0) {
        TLatex tl; tl.DrawLatex(0.3,0.5,"No data");
        continue;
      }
      h->SetLineColor(kColors[i]);
      h->SetFillColorAlpha(kColors[i], 0.25);
      h->SetFillStyle(1001);
      h->GetXaxis()->SetRangeUser(0,100);
      h->Draw("HIST");
      TLatex tl; tl.SetNDC(); tl.SetTextFont(42); tl.SetTextSize(0.055);
      tl.DrawLatex(0.15,0.92, kMediaLabels[i].c_str());
      // PPO and bis-MSB reference lines
      TLine* l1 = new TLine(2.5,0,2.5,h->GetMaximum()*0.9);
      l1->SetLineColor(kBlue); l1->SetLineStyle(2); l1->Draw();
      TLine* l2 = new TLine(12.,0,12.,h->GetMaximum()*0.7);
      l2->SetLineColor(kOrange+7); l2->SetLineStyle(3); l2->Draw();
    }
    // Global title
    c3->cd(0);
    TLatex tl; tl.SetNDC(); tl.SetTextFont(42); tl.SetTextSize(0.025);
    tl.DrawLatex(0.30,0.97,"Photon Arrival Time @ 5 MeV  |  dashed=PPO fast (2.5 ns)  dotted=bis-MSB (12 ns)");
    c3->SaveAs("root_plots/C3_Timing.pdf");
    c3->SaveAs("root_plots/C3_Timing.png");
  }

  // ── Canvas 4: Wavelength spectra overlay at 5 MeV ─────────────────────────
  {
    TCanvas* c4 = new TCanvas("c4","Wavelength",900,600);
    TLegend* leg = new TLegend(0.55,0.55,0.93,0.92);
    leg->SetBorderSize(0); leg->SetFillStyle(0);
    bool first=true;
    for (size_t i=0; i<kMedia.size(); ++i) {
      TH1D* h=nullptr;
      LoadWavelengthHist(rDir, kMedia[i], 5.0, h);
      if (!h||h->GetEntries()==0) continue;
      double mx=h->GetMaximum(); if(mx>0) h->Scale(1./mx);
      h->SetLineColor(kColors[i]); h->SetLineWidth(2);
      h->GetXaxis()->SetRangeUser(280,600);
      h->Draw(first?"HIST":"HIST SAME"); first=false;
      leg->AddEntry(h, kMediaLabels[i].c_str(), "l");
    }
    // PMT QE overlay (approximate, scaled)
    const int nQE=20;
    double wlQE[nQE]={270,290,310,330,350,370,390,410,420,440,460,480,500,520,540,560,580,600,620,640};
    double qe[nQE]  ={.05,.10,.17,.23,.27,.29,.30,.30,.30,.29,.26,.22,.17,.13,.09,.06,.04,.02,.01,.00};
    auto gqe = new TGraph(nQE,wlQE,qe);
    gqe->SetLineColor(kBlack); gqe->SetLineStyle(2); gqe->SetLineWidth(2);
    gqe->Draw("L SAME");
    leg->AddEntry(gqe,"PMT QE (bialkali)","l");
    leg->Draw();

    TLatex tl; tl.SetNDC(); tl.SetTextFont(42); tl.SetTextSize(0.042);
    tl.DrawLatex(0.14,0.93,"Detected Photon Wavelength Spectra @ 5 MeV");
    // PPO/bis-MSB annotations
    TArrow* a1 = new TArrow(360,0.85,360,0.98,0.015,">");
    a1->SetLineColor(kAzure); a1->Draw();
    tl.SetNDC(false); tl.SetTextSize(14); tl.SetTextColor(kAzure);
    tl.DrawLatex(365,0.88,"PPO emit");
    TArrow* a2 = new TArrow(420,0.75,420,0.95,0.015,">");
    a2->SetLineColor(kOrange+7); a2->Draw();
    tl.SetTextColor(kOrange+7);
    tl.DrawLatex(425,0.78,"bis-MSB emit");

    c4->SaveAs("root_plots/C4_Wavelength.pdf");
    c4->SaveAs("root_plots/C4_Wavelength.png");
  }

  // ── Canvas 5: Bar chart – detected photons at 5 MeV ──────────────────────
  {
    TCanvas* c5 = new TCanvas("c5","Bar5MeV",1100,600);
    const int n=kMedia.size();
    double vals[8]={0}, cerenk[8]={0}, scintv[8]={0};
    for (int i=0;i<n;++i) {
      auto& s=allData[kMedia[i]][5.0];
      if(s.empty()) continue;
      vals[i]  = s["mean_detected_per_event"];
      double tot = s.count("total_detected") ? s["total_detected"] : 1.;
      cerenk[i] = (tot>0 ? s["cerenkov_detected"]/tot : 0.) * vals[i];
      scintv[i] = (tot>0 ? s["scint_detected"]   /tot : 0.) * vals[i];
    }
    auto h_total  = new TH1D("hTot5", ";Medium;Photons/Event", n,0,n);
    auto h_cerenk = new TH1D("hCer5", ";Medium;",              n,0,n);
    auto h_scint  = new TH1D("hSci5", ";Medium;",              n,0,n);
    for(int i=0;i<n;++i){
      h_total->SetBinContent (i+1,vals[i]);
      h_cerenk->SetBinContent(i+1,cerenk[i]);
      h_scint->SetBinContent (i+1,scintv[i]);
      h_total->GetXaxis()->SetBinLabel(i+1, kMediaLabels[i].c_str());
    }
    h_total->SetFillColorAlpha(kGray+1,0.4);
    h_total->SetLineColor(kGray+2);
    h_cerenk->SetFillColorAlpha(kBlue,0.7);
    h_cerenk->SetLineColor(kBlue+1);
    h_scint->SetFillColorAlpha(kOrange+7,0.7);
    h_scint->SetLineColor(kOrange+8);

    h_total->Draw("BAR2");
    h_cerenk->Draw("BAR2 SAME");
    h_scint->Draw("BAR2 SAME");
    TLegend* leg=new TLegend(0.65,0.72,0.93,0.92);
    leg->SetBorderSize(0); leg->SetFillStyle(0);
    leg->AddEntry(h_total, "Total detected","f");
    leg->AddEntry(h_cerenk,"Cherenkov","f");
    leg->AddEntry(h_scint, "Scintillation (PPO+bis-MSB)","f");
    leg->Draw();
    TLatex tl; tl.SetNDC(); tl.SetTextFont(42); tl.SetTextSize(0.042);
    tl.DrawLatex(0.14,0.93,"Detected Photons/Event @ 5 MeV — All Media");
    c5->SaveAs("root_plots/C5_Bar5MeV.pdf");
    c5->SaveAs("root_plots/C5_Bar5MeV.png");
  }

  std::cout << "\n[AnalyzeRun] ROOT plots written to ./root_plots/\n";
}
