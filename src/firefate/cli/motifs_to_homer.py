"""Convert a CisBP/MEME motif file to HOMER format.

Run as: python -m firefate.cli.motifs_to_homer IN.meme OUT.motif
"""
from __future__ import annotations

import argparse

import os
import re


def parse_cisBP_motifs(file_content):
    motifs = []
    current_motif = None
    lines = file_content.split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("MOTIF"):
            # Start new motif
            if current_motif:
                motifs.append(current_motif)
            current_motif = {
                "name": line.split()[1],  # Get motif ID after "MOTIF"
                "matrix": [],
                "w": None,
            }
        elif line.startswith("letter-probability matrix"):
            if current_motif:
                # Extract width from the line
                w_match = re.search(r"w=\s*(\d+)", line)
                current_motif["w"] = int(w_match.group(1)) if w_match else None
        elif current_motif and current_motif["w"] is not None:
            # Parse probability lines (skip empty lines and URL lines)
            if line and not line.startswith("URL"):
                try:
                    probs = [float(x) for x in line.split()]
                    if len(probs) == 4:  # Check for 4 probabilities (A,C,G,T)
                        current_motif["matrix"].append(probs)
                except ValueError:
                    continue

    # Don't forget the last motif
    if current_motif and current_motif["matrix"]:
        motifs.append(current_motif)

    # Debug print
    print(f"Parsed {len(motifs)} motifs")
    for i, motif in enumerate(motifs, 1):
        print(
            f"Motif {i}: {motif['name']}, width={motif['w']}, matrix rows={len(motif['matrix'])}"
        )

    return motifs


def process_motif_file_in_homer_format(input_file, output_file=None):
    with open(input_file, "r") as f:
        content = f.read()
    motifs = parse_cisBP_motifs(content)
    print(f"Found {len(motifs)} motifs")  # Debug print
    
    # Set default output file if none provided
    if output_file is None:
        output_file = input_file + ".motif"
    
    # Track seen TF names to skip duplicates
    seen_tfs = set()  # Use a set to track unique TF names
    
    # Constant score for all motifs
    CONSTANT_SCORE = 6.9
        
    with open(output_file, "w") as f:
        for i, motif in enumerate(motifs, 1):
            # Skip if we've seen this TF before
            base_name = motif['name']
            if base_name in seen_tfs:
                print(f"Skipping duplicate motif for {base_name}")
                continue
                
            seen_tfs.add(base_name)  # Add to seen set
            
            # Debug prints
            print(f"Processing motif {i}: {base_name}")
            print(
                f"Matrix size: {len(motif['matrix'])}x{len(motif['matrix'][0]) if motif['matrix'] else 0}"
            )
            if not motif["matrix"]:
                print(f"Skipping motif {i}: empty matrix")
                continue
                
            # Calculate only consensus sequence
            consensus = "".join(["ACGT"[row.index(max(row))] for row in motif["matrix"]])
            
            # Add _MEME1 suffix
            modified_name = f"{base_name}_MEME1"
            
            # Write header line
            header = f">{consensus}\t{modified_name}\t{CONSTANT_SCORE:.5f}\t-10\n"
            f.write(header)
            print(f"Wrote header: {header.strip()}")  # Debug print
            
            # Write probability matrix
            for row in motif["matrix"]:
                line = "\t".join(map(str, row)) + "\n"
                f.write(line)
                print(f"Wrote matrix row: {line.strip()}")  # Debug print
            
            # Add blank line between motifs
            f.write("\n")
            # Ensure writing to disk
            f.flush()
    
    print(f"\nProcessed {len(seen_tfs)} unique TFs")
    print(f"Skipped {len(motifs) - len(seen_tfs)} duplicate motifs")
    
    # Verify file was written
    if os.path.exists(output_file):
        print(f"File created successfully at: {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
    else:
        print("Error: File was not created!")
    return output_file


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a CisBP/MEME motif file to HOMER format, one entry per TF."
    )
    parser.add_argument("input_file", help="CisBP/MEME motif file to read.")
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help="Destination .motif file. Default: <input_file>.motif",
    )
    args = parser.parse_args(argv)
    return process_motif_file_in_homer_format(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
