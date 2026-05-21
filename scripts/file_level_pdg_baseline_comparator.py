"""
File-Level PDG Baseline Comparator
===================================

OBJECTIVE:
---------
Extract metrics from file-level PDGs directly and compare them with baseline metrics
(which were obtained via task-level PDG extraction + file-level aggregation).

DESIGN DECISIONS & ARCHITECTURE:
-------------------------------

1. LEGACY CODE REUSE:
   - Import from: legacy.task_metrics (for metric calculation functions)
   - Key functions reused:
     * verticesCount(G) - counts nodes in PDG
     * edgesCount(G) - counts edges in PDG
     * edgesToVerticesRatio(G) - ratio calculation
     * globalInput(G) - non-local variables + parameters (requires node attributes)
     * globalOutput(G) - non-local variables modified (requires node attributes)
   
2. TASK-LEVEL vs FILE-LEVEL PDG DIFFERENCES:
   - TASK-LEVEL PDG (baseline):
     * Format: GraphML (.gml files)
     * Contains detailed node attributes: node_type, scope_level, version, location
     * FanIn/FanOut metrics are cross-playbook dependencies
     * Located: output/repositories/<repo>/<filepath>_<nodeId>.gml
     * Multiple PDGs per file (one per task), aggregated to file-level
   
   - FILE-LEVEL PDG (this script):
     * Format: DOT (pdg.dot files) - GraphQL/Graphviz format
     * Contains simplified node structure (task names + arguments)
     * NO node_type, scope_level, version attributes in DOT
     * NO cross-playbook relationships available
     * Located: output/pdg/<repository>/<commit>/<filepath>/PDG_FILE_LEVEL/pdg.dot
     * One PDG per file

3. METRIC EXTRACTION STRATEGY:
   - File-level PDG DOT → NetworkX MultiDiGraph (via nx.read_dot())
   - Metrics from simple graph structure:
     ✓ verticesCount = len(G.nodes)
     ✓ edgesCount = len(G.edges)
     ✓ edgesToVerticesRatio = edgesCount / verticesCount
     ✓ maxPdgVertices = verticesCount (since single PDG per file)
   
   - Metrics requiring node attributes (not available in file-level DOT):
     ✗ globalInput: REQUIRES node attributes → return 0 (NO DATA)
     ✗ globalOutput: REQUIRES node attributes → return 0 (NO DATA)
   
   - Cross-playbook metrics (not applicable to single file):
     ✗ directFanIn: requires playbook dictionary → return 0
     ✗ indirectFanIn: requires playbook dictionary → return 0
     ✗ directFanOut: requires playbook dictionary → return 0
     ✗ indirectFanOut: requires playbook dictionary → return 0
   
   - Cohesion metric:
     • lackOfCohesion: shared nodes between tasks
     • For file-level: = verticesCount (no task-level decomposition in DOT)

4. ERROR HANDLING:
   - Missing PDG directories → log warning, skip row
   - Empty/missing pdg.dot → log warning, skip row
   - Corrupted DOT syntax → catch exception, log error, skip row
   - Division by zero (vertices=0) → handle gracefully with 0 ratio
   - CSV read/write errors → catch and log

5. PATHLIB USAGE:
   - Replace manual string concatenation with Path objects
   - Ensures cross-platform compatibility (Windows/Linux paths)

6. LOGGING:
   - Structured logging with timestamp, level, message
   - Log each processed PDG (success/failure)
   - Log summary statistics at end

7. INPUT/OUTPUT:
   - Input: output/extraction_status.csv (columns: repository, commit, filepath, status)
   - Output: output/baseline_metric_comparison_status.csv
   - Process ONLY rows with status == "SUCCESS"

LIMITATIONS & FUTURE IMPROVEMENTS:
---------------------------------
- globalInput/globalOutput = 0 (unavailable in DOT format)
  → Possible solution: Parse GraphML if available, fallback to DOT
- FanIn/FanOut = 0 (single-file context)
  → Possible solution: Build cross-file dependency graph for comparison
- lackOfCohesion = verticesCount (no sub-task structure in DOT)
  → More research needed on how to decompose file-level PDG

TESTING VERIFICATION:
--------------------
- Sample extracted metrics vs baseline metrics (if available)
- Visual comparison of a few matching PDGs
- Statistical analysis of metric distributions
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import networkx as nx
from typing import Dict, Tuple, Optional
import traceback
import pydot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('file_level_pdg_extraction.log')
    ]
)
logger = logging.getLogger(__name__)


class FileLevelPDGExtractor:
    """Extract metrics from file-level PDGs in DOT format."""
    
    def __init__(self, workspace_root: Path):
        """
        Initialize extractor with workspace paths.
        
        Args:
            workspace_root: Root path of the project
        """
        self.workspace_root = Path(workspace_root)
        self.extraction_status_path = self.workspace_root / "output" / "extraction_status.csv"
        self.pdg_output_dir = self.workspace_root / "output" / "pdg"
        self.output_csv = self.workspace_root / "output" / "baseline_metric_comparison_status.csv"
        
        logger.info(f"Initialized extractor with workspace: {self.workspace_root}")
    
    def load_extraction_status(self) -> pd.DataFrame:
        """
        Load extraction status CSV.
        
        Returns:
            DataFrame with columns: repository, commit, filepath, status, etc.
        """
        try:
            df = pd.read_csv(self.extraction_status_path)
            logger.info(f"Loaded extraction_status.csv with {len(df)} rows")
            return df
        except FileNotFoundError:
            logger.error(f"extraction_status.csv not found at {self.extraction_status_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading extraction_status.csv: {e}")
            raise
    
    def get_pdg_file_level_path(self, repository: str, commit: str, filepath: str) -> Path:
        """
        Construct path to file-level PDG directory.
        
        Args:
            repository: Repository name (e.g., 'florianutz/Ubuntu1804-CIS')
            commit: Commit hash
            filepath: File path within repo (e.g., 'handlers/main.yml')
        
        Returns:
            Path to PDG_FILE_LEVEL directory
        """
        # Extract org/repo from full repository path if needed
        if '/' in repository:
            org, repo_name = repository.split('/', 1)
        else:
            repo_name = repository
        
        pdg_path = (
            self.pdg_output_dir /
            org /
            repo_name /
            commit /
            filepath /
            "PDG_FILE_LEVEL"
        )
        return pdg_path
    
    def load_pdg_dot(self, pdg_dir: Path) -> Optional[nx.MultiDiGraph]:
        """
        Load PDG from DOT file format.
        
        Args:
            pdg_dir: Path to PDG_FILE_LEVEL directory
        
        Returns:
            NetworkX MultiDiGraph or None if loading fails
        """
        dot_file = pdg_dir / "pdg.dot"
        
        if not dot_file.exists():
            logger.warning(f"PDG DOT file not found: {dot_file}")
            return None
        
        try:
            # Read DOT file using pydot
            graphs = pydot.graph_from_dot_file(str(dot_file))
            if not graphs:
                logger.warning(f"No graphs found in DOT file: {dot_file}")
                return None
            
            pydot_graph = graphs[0]
            
            # Create NetworkX MultiDiGraph
            G = nx.MultiDiGraph()
            
            # Add nodes from pydot graph
            for node in pydot_graph.get_nodes():
                node_name = node.get_name()
                # Skip top-level graph node
                if node_name not in ['node', 'edge', 'graph']:
                    G.add_node(node_name)
            
            # Add edges from pydot graph
            for edge in pydot_graph.get_edges():
                src = edge.get_source()
                dst = edge.get_destination()
                if src and dst:
                    G.add_edge(src, dst)
            
            logger.debug(f"Loaded PDG from {dot_file}: {len(G.nodes)} nodes, {len(G.edges)} edges")
            return G
        
        except Exception as e:
            logger.error(f"Error parsing DOT file {dot_file}: {e}\n{traceback.format_exc()}")
            return None
    
    def extract_metrics(self, G: nx.MultiDiGraph) -> Dict[str, float]:
        """
        Extract metrics from file-level PDG.
        
        Args:
            G: NetworkX MultiDiGraph representing the PDG
        
        Returns:
            Dictionary with metric names and values
        """
        metrics = {}
        
        try:
            # Basic graph metrics
            vertices_count = len(G.nodes)
            edges_count = len(G.edges)
            
            metrics['verticesCount'] = vertices_count
            metrics['edgesCount'] = edges_count
            
            # Ratio calculation with zero-division handling
            if vertices_count > 0:
                metrics['edgesToVerticesRatio'] = edges_count / vertices_count
            else:
                metrics['edgesToVerticesRatio'] = 0.0
            
            # Max vertices: for single PDG, same as vertices count
            metrics['maxPdgVertices'] = vertices_count
            
            # Cohesion: for file-level PDG without task decomposition, use vertex count
            # (In task-level, this would be shared vertices across multiple task PDGs)
            metrics['lackOfCohesion'] = vertices_count
            
            # Metrics not available in file-level DOT (no node attributes)
            # These require node_type, scope_level, version attributes
            metrics['globalInput'] = 0.0
            metrics['globalOutput'] = 0.0
            
            # Cross-playbook metrics (not applicable to single file)
            # These require comparison with other files
            metrics['directFanIn'] = 0.0
            metrics['indirectFanIn'] = 0.0
            metrics['directFanOut'] = 0.0
            metrics['indirectFanOut'] = 0.0
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error extracting metrics: {e}\n{traceback.format_exc()}")
            return {}
    
    def process_extraction_status(self) -> pd.DataFrame:
        """
        Process extraction status and extract file-level PDG metrics.
        
        Returns:
            DataFrame with extracted metrics
        """
        df_status = self.load_extraction_status()
        
        # Filter for SUCCESS rows only
        df_success = df_status[df_status['status'] == 'SUCCESS'].copy()
        logger.info(f"Processing {len(df_success)} successful extractions")
        
        results = []
        success_count = 0
        failed_count = 0
        
        for idx, row in df_success.iterrows():
            repository = row['repository']
            commit = row['commit']
            filepath = row['filepath']
            
            try:
                logger.debug(f"Processing: {repository} | {commit} | {filepath}")
                
                # Get PDG path
                pdg_dir = self.get_pdg_file_level_path(repository, commit, filepath)
                
                if not pdg_dir.exists():
                    logger.warning(f"PDG directory not found: {pdg_dir}")
                    failed_count += 1
                    continue
                
                # Load PDG
                G = self.load_pdg_dot(pdg_dir)
                if G is None:
                    failed_count += 1
                    continue
                
                # Extract metrics
                metrics = self.extract_metrics(G)
                if not metrics:
                    failed_count += 1
                    continue
                
                # Build result row
                result_row = {
                    'repository': repository,
                    'commit': commit,
                    'filepath': filepath,
                }
                result_row.update(metrics)
                results.append(result_row)
                success_count += 1
                
                if success_count % 100 == 0:
                    logger.info(f"Processed {success_count} PDGs successfully")
            
            except Exception as e:
                logger.error(
                    f"Unexpected error processing {repository}/{commit}/{filepath}: {e}\n{traceback.format_exc()}"
                )
                failed_count += 1
                continue
        
        logger.info(f"Extraction complete: {success_count} successful, {failed_count} failed")
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        return df_results
    
    def save_results(self, df_results: pd.DataFrame) -> None:
        """
        Save extraction results to CSV.
        
        Args:
            df_results: DataFrame with extracted metrics
        """
        try:
            # Ensure output directory exists
            self.output_csv.parent.mkdir(parents=True, exist_ok=True)
            
            # Define column order
            columns_order = [
                'repository', 'commit', 'filepath',
                'maxPdgVertices', 'verticesCount', 'edgesToVerticesRatio', 'edgesCount',
                'globalInput', 'lackOfCohesion',
                'indirectFanOut', 'indirectFanIn',
                'directFanOut', 'directFanIn',
                'globalOutput'
            ]
            
            # Reorder columns
            df_results = df_results[[col for col in columns_order if col in df_results.columns]]
            
            # Save to CSV
            df_results.to_csv(self.output_csv, index=False)
            logger.info(f"Results saved to {self.output_csv} ({len(df_results)} rows)")
        
        except Exception as e:
            logger.error(f"Error saving results: {e}\n{traceback.format_exc()}")
            raise
    
    def run(self) -> None:
        """Run the full extraction pipeline."""
        logger.info("="*80)
        logger.info("Starting File-Level PDG Baseline Comparator")
        logger.info("="*80)
        
        try:
            df_results = self.process_extraction_status()
            self.save_results(df_results)
            logger.info("="*80)
            logger.info("Extraction completed successfully!")
            logger.info("="*80)
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
            sys.exit(1)


def main():
    """Main entry point."""
    # Determine workspace root
    workspace_root = Path(__file__).parent.parent
    
    extractor = FileLevelPDGExtractor(workspace_root)
    extractor.run()


if __name__ == "__main__":
    main()
