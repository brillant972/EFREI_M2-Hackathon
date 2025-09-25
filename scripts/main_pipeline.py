def main():
    """Orchestrateur principal du pipeline"""
    
    print("PIPELINE COMPLET - PRÉDICTION INONDATIONS IDF")
    print("=" * 60)
    print("Hackathon EFREI - Ville Durable et Intelligente")
    print("Architecture Bronze-Silver-Gold avec ML")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Phase 0 : Préparation environnement
        logger.info("Phase 0 : Préparation environnement")
        create_project_structure()
        create_requirements()
        
        # Skipping dependency check for now
        # if not check_dependencies():
        #     logger.error("Veuillez installer les dépendances manquantes")
        #     return False
        
        # Phase 1 : Exécution Bronze
        logger.info("Phase 1 : Couche Bronze (Ingestion)")
        bronze_success, bronze_result = run_script("bronze", "Ingestion données brutes")
        if not bronze_success:
            logger.error("Échec couche Bronze")
            return False
        
        # Phase 2 : Exécution Silver  
        logger.info("Phase 2 : Couche Silver (Feature Engineering)")
        silver_success, silver_result = run_script("silver", "Nettoyage et feature engineering")
        if not silver_success:
            logger.error("Échec couche Silver")
            return False
        
        # Phase 3 : Exécution Gold
        logger.info("Phase 3 : Couche Gold (Machine Learning)")
        gold_success, gold_result = run_script("gold", "Modélisation ML et prédictions")
        if not gold_success:
            logger.error("Échec couche Gold") 
            return False
        
        # Phase 4 : Génération dashboard
        logger.info("Phase 4 : Génération dashboard")
        generate_dashboard_config()
        
        # Phase 5 : Résumé final
        logger.info("Phase 5 : Génération résumé")
        generate_execution_summary(bronze_result, silver_result, gold_result)
        
        # Statistiques finales
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\nPIPELINE TERMINÉ AVEC SUCCÈS !")
        print("=" * 60)
        print(f"Durée totale : {duration}")
        print(f"Toutes les phases exécutées avec succès")
        print(f"\nFichiers générés :")
        print(f"   • data/ : Données Bronze-Silver-Gold")
        print(f"   • models/ : Modèles ML entraînés")
        print(f"   • visualizations/ : Graphiques et cartes")
        print(f"   • reports/ : Rapport final hackathon")
        print(f"   • dashboard.py : Interface Streamlit")
        print(f"\nCOMMANDES DE DÉMONSTRATION :")
        print(f"   streamlit run dashboard.py")
        print(f"   # Puis ouvrir : http://localhost:8501")
        print(f"\nPRÊT POUR LA PRÉSENTATION HACKATHON !")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur critique dans pipeline principal: {e}")
        return False